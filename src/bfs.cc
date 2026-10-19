// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "gacge.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in the parent array as negative numbers. Thus, the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

// GACGE: the bottom-up step's chunks are ranges of vertices cut once per graph
// (BUBounds, outside the trials) by both their in-edges (at most
// GACGE_BFS_BU_EDGES, default 262144, at least 16 ranges a thread and 4096
// edges a range) and their vertices (at most 1024 or a sixteenth of a thread's
// share); 0: GAP's 1024-vertex chunks. A vertex's work is its in-edges while
// most are unvisited (vertex-count chunks of a web graph's dense rows ran long
// while the others waited, and 1024-vertex chunks had 96 threads contend for the
// schedule: Sinaweibo's 57K), one read of parent once visited (so the late steps
// balance by vertices). The ranges are taken dynamically.
static const int64_t kBUEdges = getenv("GACGE_BFS_BU_EDGES") ? atoll(getenv("GACGE_BFS_BU_EDGES")) : 262144;

pvector<NodeID> BUBounds(const Graph &g) {
  pvector<NodeID> bounds;
  bounds.push_back(0);
  if (kBUEdges > 0) {
    const int64_t work = g.num_edges_directed() + g.num_nodes();
    const int64_t per = std::max<int64_t>(4096, std::min(kBUEdges, work / (16 * omp_get_max_threads())));
    const int64_t vcap = std::max<int64_t>(1024, g.num_nodes() / (16 * omp_get_max_threads()));
    // cuts at multiples of 64: a bitmap word (64 vertices) is written by one
    // range only, so next.set_bit needs no atomic (as GAP's 1024-vertex chunks)
    int64_t acc = 0;
    NodeID start = 0;
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      acc += g.in_degree(u) + 1;
      if (acc >= per || u + 1 - start >= vcap) {
        const NodeID b = std::min<int64_t>(g.num_nodes(), ((int64_t)u + 64) / 64 * 64);
        bounds.push_back(b);
        acc = 0;
        start = b;
        u = b - 1;
      }
    }
  }
  if (bounds[bounds.size() - 1] != g.num_nodes())
    bounds.push_back(g.num_nodes());
  return bounds;
}

int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next, const pvector<NodeID> &bounds) {
  int64_t awake_count = 0;
  if (kBUEdges <= 0)
    next.reset();
  #pragma omp parallel
  {
    GACGE_SYNC_BEGIN(1);  // GACGE: barrier wait (gacge.h sync stats)
    if (kBUEdges > 0) {
      const int64_t nchunks = bounds.size() - 1;
      #pragma omp for reduction(+ : awake_count) schedule(dynamic, 1) GACGE_NOWAIT
      for (int64_t c = 0; c < nchunks; c++) {
        next.clear_words(bounds[c], bounds[c + 1]);  // a range's own words (GAP: next.reset() first)
        for (NodeID u = bounds[c]; u < bounds[c + 1]; u++) {
          if (parent[u] < 0) {
            for (NodeID v : g.in_neigh(u)) {
              if (front.get_bit(v)) {
                parent[u] = v;
                awake_count++;
                next.set_bit(u);
                break;
              }
            }
          }
        }
      }
    } else {
      #pragma omp for reduction(+ : awake_count) schedule(dynamic, 1024) GACGE_NOWAIT
      for (NodeID u=0; u < g.num_nodes(); u++) {
        if (parent[u] < 0) {
          for (NodeID v : g.in_neigh(u)) {
            if (front.get_bit(v)) {
              parent[u] = v;
              awake_count++;
              next.set_bit(u);
              break;
            }
          }
        }
      }
    }
    GACGE_SYNC_END(1, "bu");
  }
  return awake_count;
}


// GAP's top-down step splits the queue statically. The sync-stats build (an experiment
// only) runs it under schedule(runtime): static by default, GACGE_BFS_TD_SCHED=dynamic[,<chunk>]
// for a dynamic schedule.
#if GACGE_SYNC_STATS_BUILD
static const bool kTDSched = [] {
  const char *e = getenv("GACGE_BFS_TD_SCHED");
  int chunk = 0;
  if (e != nullptr && strncmp(e, "dynamic", 7) == 0) {
    if (e[7] == ',') chunk = atoi(e + 8);
    omp_set_schedule(omp_sched_dynamic, chunk > 0 ? chunk : 1);
  } else {
    omp_set_schedule(omp_sched_static, 0);
  }
  return true;
}();
#endif

int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    GACGE_SYNC_BEGIN(0);  // GACGE: barrier wait (gacge.h sync stats)
    QueueBuffer<NodeID> lqueue(queue);
#if GACGE_SYNC_STATS_BUILD
    #pragma omp for reduction(+ : scout_count) nowait schedule(runtime)
#else
    #pragma omp for reduction(+ : scout_count) nowait
#endif
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;
      for (NodeID v : g.out_neigh(u)) {
        NodeID curr_val = parent[v];
        if (curr_val < 0) {
          if (compare_and_swap(parent[v], curr_val, u)) {
            lqueue.push_back(v);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
    GACGE_SYNC_END(0, "td");
  }
  return scout_count;
}


// GACGE: the top-down step edge-balanced (the default; GACGE_BFS_TD=vertex is
// GAP's TDStep above, which splits the queue's vertices over the threads, so a
// hub's edges are one thread's: busy 7-9% of 96 threads on Arabic, Kron21 and
// Sinaweibo). A prefix sum of the frontier's degrees splits its out-edges into
// ranges of about equal size, taken dynamically, so a hub's edges are spread
// over the threads as the GPU's merge-path advance spreads them. Same
// relaxation (a claim by compare-and-swap, the thread-local queues), same work.
static const bool kTDEdges = [] {
  const char *e = getenv("GACGE_BFS_TD");
  return e == nullptr || strcmp(e, "vertex") != 0;
}();

// a frontier below GACGE_BFS_TD_SERIAL entries (default 1024) has its prefix sum by one
// thread (a degree is a random read of the offsets): one barrier instead of three
static const int64_t kTDSerial = getenv("GACGE_BFS_TD_SERIAL") ? atoll(getenv("GACGE_BFS_TD_SERIAL")) : 1024;

int64_t TDStepEdges(const Graph &g, pvector<NodeID> &parent, SlidingQueue<NodeID> &queue,
                    pvector<int64_t> &offs, pvector<int64_t> &sums) {
  const NodeID *q = queue.begin();
  const int64_t n = queue.size();
  int64_t scout_count = 0;
  int64_t total = 0;
  #pragma omp parallel
  {
    GACGE_SYNC_BEGIN(0);  // GACGE: barrier wait (gacge.h sync stats)
    // offs[i]: the frontier's edges before entry i; offs[n] = total. A small
    // frontier's by one thread (one barrier), a large one's by all (three)
    if (n < kTDSerial) {
      #pragma omp single
      {
        int64_t acc = 0;
        for (int64_t i = 0; i < n; i++) {
          offs[i] = acc;
          acc += g.out_degree(q[i]);
        }
        offs[n] = total = acc;
      }
    } else {
      const int P = omp_get_num_threads(), t = omp_get_thread_num();
      const int64_t lo = n * t / P, hi = n * (t + 1) / P;
      int64_t s = 0;
      for (int64_t i = lo; i < hi; i++)
        s += g.out_degree(q[i]);
      sums[t] = s;
      #pragma omp barrier
      int64_t base = 0;
      for (int i = 0; i < t; i++)
        base += sums[i];
      for (int64_t i = lo; i < hi; i++) {
        offs[i] = base;
        base += g.out_degree(q[i]);
      }
      #pragma omp single
      {
        for (int i = 0; i < P; i++)
          total += sums[i];
        offs[n] = total;
      }
    }
    const int P = omp_get_num_threads();
    // ranges of about 512 edges, at most 16 a thread, taken dynamically
    const int64_t nchunks = std::max<int64_t>(1, std::min<int64_t>((total + 511) / 512, (int64_t)P * 16));
    const int64_t chunk = (total + nchunks - 1) / nchunks;
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for reduction(+ : scout_count) nowait schedule(dynamic, 1)
    for (int64_t c = 0; c < nchunks; c++) {
      const int64_t e0 = c * chunk, e1 = std::min(total, e0 + chunk);
      if (e0 >= e1)
        continue;
      // the first entry with edges in [e0, e1): the last i with offs[i] <= e0
      int64_t i = std::upper_bound(offs.begin(), offs.begin() + n, e0) - offs.begin() - 1;
      for (; i < n && offs[i] < e1; i++) {
        const NodeID u = q[i];
        const NodeID *nb = g.out_neigh(u).begin();
        const int64_t a = std::max(e0, offs[i]) - offs[i], b = std::min(e1, offs[i + 1]) - offs[i];
        for (int64_t k = a; k < b; k++) {
          const NodeID v = nb[k];
          NodeID curr_val = parent[v];
          if (curr_val < 0) {
            if (compare_and_swap(parent[v], curr_val, u)) {
              lqueue.push_back(v);
              scout_count += -curr_val;
            }
          }
        }
      }
    }
    lqueue.flush();
    GACGE_SYNC_END(0, "td");
  }
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

// GACGE: word by word, zero words skipped (GAP tested every bit: 1 ms over
// Sinaweibo's 58M for a frontier of 421); GACGE_BFS_B2Q=bit is GAP's loop
static const bool kB2QWords = [] {
  const char *e = getenv("GACGE_BFS_B2Q");
  return e == nullptr || strcmp(e, "bit") != 0;
}();

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    if (kB2QWords) {
      const int64_t nw = bm.num_words(), nn = g.num_nodes();
      #pragma omp for nowait schedule(dynamic, 1024)
      for (int64_t w = 0; w < nw; w++) {
        uint64_t x = bm.word(w);
        while (x != 0) {
          const int b = __builtin_ctzll(x);
          const int64_t n = w * 64 + b;
          if (n < nn)
            lqueue.push_back(n);
          x &= x - 1;
        }
      }
    } else {
      #pragma omp for nowait
      for (NodeID n=0; n < g.num_nodes(); n++)
        if (bm.get_bit(n))
          lqueue.push_back(n);
    }
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
  return parent;
}

// GACGE: DOBFS's state, allocated once outside the trials; PrepareBFS sets it
// to what DOBFS initialized at its start (InitParent, the source in the queue,
// cleared bitmaps), untimed (BenchmarkKernelPrepared).
struct BFSState {
  NodeID source = -1;
  pvector<NodeID> parent;
  SlidingQueue<NodeID> queue;
  Bitmap curr, front;
  pvector<int64_t> offs, sums;  // TDStepEdges: the frontier's edge offsets, the threads' partial sums
  pvector<NodeID> bu_bounds;    // BUStep's vertex ranges (BUBounds)
  explicit BFSState(const Graph &g) : parent(g.num_nodes()), queue(g.num_nodes()),
      curr(g.num_nodes()), front(g.num_nodes()), offs(g.num_nodes() + 1, 0),
      sums(omp_get_max_threads() + 1, 0), bu_bounds(BUBounds(g)) {}
};

void PrepareBFS(const Graph &g, NodeID source, BFSState &s) {
  s.source = source;
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    s.parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
  s.parent[source] = source;
  s.queue.reset();
  s.queue.push_back(source);
  s.queue.slide_window();
  s.curr.reset();
  s.front.reset();
}

pvector<NodeID>& DOBFS(const Graph &g, BFSState &s, bool logging_enabled = false,
                       int alpha = 15, int beta = 18) {
  const NodeID source = s.source;
  if (logging_enabled)
    PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  pvector<NodeID> &parent = s.parent;
  SlidingQueue<NodeID> &queue = s.queue;
  Bitmap &curr = s.curr;
  Bitmap &front = s.front;
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  const bool trace = gacge::Tracing();  // GACGE: one XBStep line per step (gacge.h)
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      if (logging_enabled)
        PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        if (trace) {
          int64_t fe = 0;
          #pragma omp parallel for reduction(+ : fe)
          for (NodeID u = 0; u < g.num_nodes(); u++)
            if (front.get_bit(u)) fe += g.out_degree(u);
          gacge::TraceStep("bu", "pull", awake_count, fe);
        }
        t.Start();
        old_awake_count = awake_count;
        {
          gacge::PerfPhaseRegion r("bu");  // GACGE: GACGE_PERF_PHASE=bu counts the bottom-up steps
          awake_count = BUStep(g, parent, front, curr, s.bu_bounds);
        }
        front.swap(curr);
        t.Stop();
        if (logging_enabled)
          PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      if (logging_enabled)
        PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      if (trace)
        gacge::TraceStep("td", "push", queue.size(), scout_count);
      t.Start();
      edges_to_check -= scout_count;
      {
        gacge::PerfPhaseRegion r("td");  // GACGE: GACGE_PERF_PHASE=td counts the top-down steps
        scout_count = kTDEdges ? TDStepEdges(g, parent, queue, s.offs, s.sums) : TDStep(g, parent, queue);
      }
      queue.slide_window();
      t.Stop();
      if (logging_enabled)
        PrintStep("td", t.Seconds(), queue.size());
    }
  }
  t.Start();
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    if (parent[n] < -1)
      parent[n] = -1;
  t.Stop();
  if (logging_enabled)
    PrintStep("f", t.Seconds());
  return parent;
}


void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  SourcePicker<Graph> sp(g, cli.start_vertex());
  BFSState state(g);  // GACGE: allocated once, prepared untimed per trial
  auto Prepare = [&sp, &state] (const Graph &g) {
    PrepareBFS(g, sp.PickNext(), state);
  };
  auto BFSBound = [&state, &cli] (const Graph &g) -> pvector<NodeID>& {
    // GACGE: GACGE_BFS_ALPHA / GACGE_BFS_BETA override GAP's switching thresholds (15, 18)
    static const int alpha = getenv("GACGE_BFS_ALPHA") ? atoi(getenv("GACGE_BFS_ALPHA")) : 15;
    static const int beta = getenv("GACGE_BFS_BETA") ? atoi(getenv("GACGE_BFS_BETA")) : 18;
    return DOBFS(g, state, cli.logging_en(), alpha, beta);
  };
  auto VerifierBound = [&state] (const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, state.source, parent);
  };
  BenchmarkKernelPrepared(cli, g, Prepare, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
