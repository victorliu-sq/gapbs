// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <iostream>
#include <omp.h>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "gacge.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer, Yunming Zhang

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph. This implementation
incorporates a new bucket fusion optimization [2] that significantly reduces
the number of iterations (& barriers) needed.

The bins of width delta are actually all thread-local and of type std::vector,
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies its selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and, it now appears in a lower bin. We find ignoring vertices if
their distance is less than the min distance for the current bin removes
enough redundant work to be faster than removing the vertex from older bins.

The bucket fusion optimization [2] executes the next thread-local bin in
the same iteration if the vertices in the next thread-local bin have the
same priority as those in the current shared bin. This optimization greatly
reduces the number of iterations needed without violating the priority-based
execution order, leading to significant speedup on large diameter road networks.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala,
    Shoaib Kamil, Saman Amarasinghe, and Julian Shun. "Optimizing ordered graph
    algorithms with GraphIt." The 18th International Symposium on Code Generation
    and Optimization (CGO), pages 158-170, 2020.
*/


using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const size_t kBinSizeThreshold = 1000;

// GACGE: u's out-edges [k0, k1) (the frontier's edge ranges: TDStep-like
// splitting of a hub's edges over the threads; the fusion relaxes whole vertices)
inline
void RelaxEdges(const WGraph &g, NodeID u, WeightT delta,
                pvector<WeightT> &dist, vector <vector<NodeID>> &local_bins,
                bool trace, int64_t k0, int64_t k1) {
  if (trace)  // GACGE: every relaxation, the bin fusion's included (gacge.h)
    gacge::TraceWork(k1 - k0);
  const WNode *nb = g.out_neigh(u).begin();
  for (int64_t k = k0; k < k1; k++) {
    const WNode wn = nb[k];
    WeightT old_dist = dist[wn.v];
    WeightT new_dist = dist[u] + wn.w;
    while (new_dist < old_dist) {
      if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
        size_t dest_bin = new_dist/delta;
        if (dest_bin >= local_bins.size())
          local_bins.resize(dest_bin+1);
        local_bins[dest_bin].push_back(wn.v);
        break;
      }
      old_dist = dist[wn.v];      // swap failed, recheck dist update & retry
    }
  }
}

// GACGE: DeltaStep's distances, shared frontier (one entry per directed edge)
// and the threads' bins (GAP's local_bins: allocated inside each trial and
// grown by push_back, here kept across the trials with their capacity, as the
// GPU's bins are allocated once), allocated outside the trials; PrepareSSSP
// sets them to what DeltaStep initialized at its start, untimed
// (BenchmarkKernelPrepared).
struct SSSPThread {
  vector<vector<NodeID>> bins;  // GAP's local_bins
  vector<NodeID> fused;         // a fusion round's vertices (GAP copied the bin)
  vector<int64_t> offs;         // a small frontier's edge offsets, every thread its own
};
struct SSSPState {
  NodeID source = -1;
  pvector<WeightT> dist;
  pvector<NodeID> frontier;
  vector<SSSPThread> threads;
  pvector<int64_t> offs;  // a large frontier's edge offsets (frontier[i]'s edges before it)
  pvector<int64_t> sums;  // the threads' partial sums
  explicit SSSPState(const WGraph &g) : dist(g.num_nodes()), frontier(g.num_edges_directed()),
      threads(omp_get_max_threads()), offs(g.num_edges_directed() + 1, 0),
      sums(omp_get_max_threads() + 1, 0) {}
};

void PrepareSSSP(const WGraph &g, NodeID source, SSSPState &s) {
  s.source = source;
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    s.dist[n] = kDistInf;
  s.dist[source] = 0;
  for (SSSPThread &t : s.threads) {
    for (vector<NodeID> &b : t.bins)
      b.clear();
    t.fused.clear();
  }
}

// GACGE: the relaxation loop. GACGE_SSSP_RELAX=edges: the live entries' edges in
// ranges of about equal size (a prefix sum of their degrees: a frontier below
// GACGE_SSSP_SMALL entries, default 1024, by every thread for itself, no barrier;
// a larger one by all, two barriers), so a hub's edges are spread over the
// threads; entry: GAP's loop over the entries, a hub's edges one thread's, in
// chunks of GACGE_SSSP_CHUNK entries (GAP's 64; 0, the default: at least 64 and
// a sixteenth of a thread's share, so a large bin is not 100K chunks). Default
// (auto): edges when the graph's average degree is at least GACGE_SSSP_EDGES_DEG
// (16): the prefix sum reads a degree per entry, which a low-degree graph's few
// relaxations per entry do not amortize (Sinaweibo, 9 edges a vertex: +10%).
static const int64_t kSmallFrontier = getenv("GACGE_SSSP_SMALL") ? atoll(getenv("GACGE_SSSP_SMALL")) : 1024;
static const int64_t kEntryChunk = getenv("GACGE_SSSP_CHUNK") ? atoll(getenv("GACGE_SSSP_CHUNK")) : 0;
static const int64_t kEdgesDeg = getenv("GACGE_SSSP_EDGES_DEG") ? atoll(getenv("GACGE_SSSP_EDGES_DEG")) : 16;
static bool RelaxByEdges(const WGraph &g) {
  const char *e = getenv("GACGE_SSSP_RELAX");
  if (e != nullptr && strcmp(e, "entry") == 0) return false;
  if (e != nullptr && strcmp(e, "edges") == 0) return true;
  return g.num_edges_directed() / g.num_nodes() >= kEdgesDeg;
}

// GAP's relaxation chunk: 64 frontier entries. The sync-stats build (an experiment only)
// takes GACGE_SSSP_CHUNK=<entries> instead.
#if GACGE_SYNC_STATS_BUILD
static const int kRelaxChunk = getenv("GACGE_SSSP_CHUNK") ? atoi(getenv("GACGE_SSSP_CHUNK")) : 64;
#endif

pvector<WeightT>& DeltaStep(const WGraph &g, SSSPState &s, WeightT delta,
                            bool logging_enabled = false) {
  Timer t;
  const NodeID source = s.source;
  pvector<WeightT> &dist = s.dist;
  pvector<NodeID> &frontier = s.frontier;
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
  const bool trace = gacge::Tracing();  // GACGE: one XBStep line per bin iteration (gacge.h)
  static const bool kRelaxEdges = RelaxByEdges(g);
  t.Start();
  #pragma omp parallel
  {
    const int P = omp_get_num_threads(), tid = omp_get_thread_num();
    SSSPThread &ts = s.threads[tid];
    vector<vector<NodeID> > &local_bins = ts.bins;
    size_t iter = 0;
    while (shared_indexes[iter&1] != kMaxBin) {
      size_t &curr_bin_index = shared_indexes[iter&1];
      size_t &next_bin_index = shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = frontier_tails[iter&1];
      size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
      if (trace) {
        // the frontier entries this iteration relaxes (stale ones are skipped)
        #pragma omp single
        {
          int64_t fv = 0, fe = 0;
          for (size_t i = 0; i < curr_frontier_tail; i++) {
            NodeID u = frontier[i];
            if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index)) {
              fv++;
              fe += g.out_degree(u);
            }
          }
          gacge::TraceStep("bin", "push", fv, fe);
        }
      }
      // GACGE: barrier wait (gacge.h sync stats): the bin's relaxations (to the mark) and the thread's local
      // fusion; region 1: bins of fewer entries than 64 (the chunk) x the threads, region 0 the others
#if GACGE_SYNC_STATS_BUILD
      // region 2: fewer entries than threads, 1: fewer than 64 (GAP's chunk) x threads, 0: the others
      const size_t sync_p = static_cast<size_t>(omp_get_num_threads());
      const int sync_r = curr_frontier_tail < sync_p ? 2 : curr_frontier_tail < 64 * sync_p ? 1 : 0;
#else
      const int sync_r = 0;
#endif
      (void)sync_r;
      GACGE_SYNC_BEGIN(sync_r);
      const WeightT floor_d = delta * static_cast<WeightT>(curr_bin_index);
      if (kRelaxEdges) {
        // the live entries' out-edges in ranges of about 512, at most 16 a
        // thread, taken dynamically (a stale entry, GAP's check, has none)
        const int64_t n = curr_frontier_tail;
        const int64_t *offs;
        int64_t total = 0;
        if (n < kSmallFrontier) {
          ts.offs.resize(n + 1);
          for (int64_t i = 0; i < n; i++) {
            ts.offs[i] = total;
            const NodeID u = frontier[i];
            if (dist[u] >= floor_d)
              total += g.out_degree(u);
          }
          ts.offs[n] = total;
          offs = ts.offs.data();
        } else {
          const int64_t lo = n * tid / P, hi = n * (tid + 1) / P;
          int64_t part = 0;
          for (int64_t i = lo; i < hi; i++) {
            const NodeID u = frontier[i];
            if (dist[u] >= floor_d)
              part += g.out_degree(u);
          }
          s.sums[tid] = part;
          #pragma omp barrier
          int64_t base = 0;
          for (int i = 0; i < tid; i++)
            base += s.sums[i];
          for (int64_t i = lo; i < hi; i++) {
            s.offs[i] = base;
            const NodeID u = frontier[i];
            if (dist[u] >= floor_d)
              base += g.out_degree(u);
          }
          if (tid == P - 1)
            s.offs[n] = base;
          for (int i = 0; i < P; i++)
            total += s.sums[i];
          #pragma omp barrier
          offs = s.offs.data();
        }
        const int64_t nchunks = max<int64_t>(1, min<int64_t>((total + 511) / 512, (int64_t)P * 16));
        const int64_t chunk = (total + nchunks - 1) / nchunks;
        #pragma omp for nowait schedule(dynamic, 1)
        for (int64_t c = 0; c < nchunks; c++) {
          const int64_t e0 = c * chunk, e1 = min(total, e0 + chunk);
          if (e0 >= e1)
            continue;
          int64_t i = upper_bound(offs, offs + n, e0) - offs - 1;
          for (; i < n && offs[i] < e1; i++) {
            if (offs[i + 1] == offs[i])
              continue;
            const NodeID u = frontier[i];
            const int64_t a = max(e0, offs[i]) - offs[i], b = min(e1, offs[i + 1]) - offs[i];
            GACGE_SYNC_ITEM(sync_r, b - a + 1.0);  // GACGE: the range's edges
            RelaxEdges(g, u, delta, dist, local_bins, trace, a, b);
          }
        }
      } else {
#if GACGE_SYNC_STATS_BUILD
        const int64_t chunk = kRelaxChunk;
#else
        const int64_t chunk = kEntryChunk > 0 ? kEntryChunk
            : max<int64_t>(64, (int64_t)curr_frontier_tail / (16 * P));
#endif
        #pragma omp for nowait schedule(dynamic, chunk)
        for (size_t i=0; i < curr_frontier_tail; i++) {
          NodeID u = frontier[i];
          if (dist[u] >= floor_d) {
            GACGE_SYNC_ITEM(sync_r, g.out_degree(u) + 1.0);  // GACGE: the entry's edges (stale ones do none)
            RelaxEdges(g, u, delta, dist, local_bins, trace, 0, g.out_degree(u));
          }
        }
      }
      GACGE_SYNC_MARK(sync_r);
      while (curr_bin_index < local_bins.size() &&
             !local_bins[curr_bin_index].empty() &&
             local_bins[curr_bin_index].size() < kBinSizeThreshold) {
        // the bin's vertices, taken whole (GAP copied them; the swap keeps the capacities)
        ts.fused.swap(local_bins[curr_bin_index]);
        local_bins[curr_bin_index].clear();
        for (NodeID u : ts.fused)
          RelaxEdges(g, u, delta, dist, local_bins, trace, 0, g.out_degree(u));
        ts.fused.clear();
      }
      for (size_t i=curr_bin_index; i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
          // next_bin_index = min(next_bin_index, i), by compare-and-swap (GAP: omp critical)
          size_t seen = next_bin_index;
          while (i < seen && !compare_and_swap(next_bin_index, seen, i))
            seen = next_bin_index;
          break;
        }
      }
      GACGE_SYNC_END(sync_r, sync_r == 2 ? "bin_tiny" : sync_r == 1 ? "bin_small" : "bin");
      #pragma omp barrier
      #pragma omp single nowait
      {
        t.Stop();
        if (logging_enabled)
          PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
        t.Start();
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      #pragma omp barrier
    }
    #pragma omp single
    if (logging_enabled)
      cout << "took " << iter << " iterations" << endl;
  }
  return dist;
}


void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}


int main(int argc, char* argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  SourcePicker<WGraph> sp(g, cli.start_vertex());
  SSSPState state(g);  // GACGE: allocated once, prepared untimed per trial
  auto Prepare = [&sp, &state] (const WGraph &g) {
    PrepareSSSP(g, sp.PickNext(), state);
  };
  auto SSSPBound = [&state, &cli] (const WGraph &g) -> pvector<WeightT>& {
    return DeltaStep(g, state, cli.delta(), cli.logging_en());
  };
  auto VerifierBound = [&state] (const WGraph &g, const pvector<WeightT> &dist) {
    return SSSPVerifier(g, state.source, dist);
  };
  BenchmarkKernelPrepared(cli, g, Prepare, SSSPBound, PrintSSSPStats, VerifierBound);
  return 0;
}
