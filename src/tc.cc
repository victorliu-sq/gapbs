// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// Encourage use of gcc's parallel algorithms (for sort for relabeling)
#ifdef _OPENMP
  #define _GLIBCXX_PARALLEL
#endif

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <parallel/algorithm>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"


/*
GAP Benchmark Suite
Kernel: Triangle Counting (TC)
Author: Scott Beamer

Will count the number of triangles (cliques of size 3)

Input graph requirements:
  - undirected
  - has no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors are sorted.

This implementation relabels the vertices by degree. This optimization is
beneficial if the average degree is sufficiently high and if the degree
distribution is sufficiently non-uniform. To decide whether to relabel the
graph, we use the heuristic in WorthRelabelling.
*/


using namespace std;

size_t OrderedCount(const Graph &g) {
  size_t total = 0;
  #pragma omp parallel
  {
    GACGE_SYNC_BEGIN(0);  // GACGE: barrier wait (gacge.h sync stats)
    #pragma omp for reduction(+ : total) schedule(dynamic, 64) GACGE_NOWAIT
    for (NodeID u=0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u)) {
        if (v > u)
          break;
        auto it = g.out_neigh(v).begin();
        for (NodeID w : g.out_neigh(u)) {
          if (w > v)
            break;
          while (*it < w)
            it++;
          if (w == *it)
            total++;
        }
      }
    }
    GACGE_SYNC_END(0, "tc");
  }
  return total;
}


// Heuristic to see if sufficiently dense power-law graph
bool WorthRelabelling(const Graph &g) {
  int64_t average_degree = g.num_edges() / g.num_nodes();
  if (average_degree < 10)
    return false;
  SourcePicker<Graph> sp(g);
  int64_t num_samples = min(int64_t(1000), g.num_nodes());
  int64_t sample_total = 0;
  pvector<int64_t> samples(num_samples);
  for (int64_t trial=0; trial < num_samples; trial++) {
    samples[trial] = g.out_degree(sp.PickNext());
    sample_total += samples[trial];
  }
  sort(samples.begin(), samples.end());
  double sample_average = static_cast<double>(sample_total) / num_samples;
  double sample_median = samples[num_samples/2];
  return sample_average / 1.3 > sample_median;
}


// Uses heuristic to see if worth relabeling
// GACGE: GAP's RelabelByDegree sorted the (degree, id) pairs with a serial
// std::sort and allocated the relabelled graph inside every trial (Arabic:
// 1.17 s of 2.63; UK-2002: 68% of the time; the GPU sorts on the device from
// buffers it keeps). The same relabelling (the pairs in decreasing (degree, id)
// order, each new list sorted) with a parallel sort and buffers allocated once
// (the first, warm-up trial), then GAP's OrderedCount over the new lists.
// GACGE_TC_RELABEL=gap is GAP's.
static const bool kRelabelGap = getenv("GACGE_TC_RELABEL") && strcmp(getenv("GACGE_TC_RELABEL"), "gap") == 0;

struct RelabelScratch {
  int64_t n = 0;
  pvector<std::pair<int64_t, NodeID>> pairs;
  pvector<NodeID> new_ids;
  pvector<SGOffset> offsets;
  pvector<int64_t> sums;
  NodeID *neighs = nullptr;
  NodeID **index = nullptr;
};
static RelabelScratch rs;

size_t OrderedCountLists(int64_t n, NodeID **index) {
  size_t total = 0;
  #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
  for (NodeID u = 0; u < n; u++) {
    for (const NodeID *vp = index[u]; vp < index[u + 1]; vp++) {
      const NodeID v = *vp;
      if (v > u)
        break;
      const NodeID *it = index[v];
      for (const NodeID *wp = index[u]; wp < index[u + 1]; wp++) {
        const NodeID w = *wp;
        if (w > v)
          break;
        while (*it < w)
          it++;
        if (w == *it)
          total++;
      }
    }
  }
  return total;
}

size_t OrderedCountRelabelled(const Graph &g) {
  Timer t;
  t.Start();
  const int64_t n = g.num_nodes(), m = g.num_edges_directed();
  const int P = omp_get_max_threads();
  if (rs.n != n) {
    rs.pairs.resize(n);
    rs.new_ids.resize(n);
    rs.offsets.resize(n + 1);
    rs.sums.resize(P + 1);
    delete[] rs.neighs;
    delete[] rs.index;
    rs.neighs = new NodeID[m];
    rs.index = new NodeID*[n + 1];
    #pragma omp parallel for
    for (int64_t e = 0; e < m; e++)
      rs.neighs[e] = 0;
    rs.n = n;
  }
  #pragma omp parallel for
  for (NodeID v = 0; v < n; v++)
    rs.pairs[v] = std::make_pair(g.out_degree(v), v);
  __gnu_parallel::sort(rs.pairs.begin(), rs.pairs.end(), std::greater<std::pair<int64_t, NodeID>>());
  // offsets: a prefix sum of the new degrees; new_ids: old -> new
  #pragma omp parallel
  {
    const int t = omp_get_thread_num();
    const int64_t lo = n * t / P, hi = n * (t + 1) / P;
    int64_t s = 0;
    for (int64_t i = lo; i < hi; i++) {
      rs.new_ids[rs.pairs[i].second] = i;
      s += rs.pairs[i].first;
    }
    rs.sums[t] = s;
    #pragma omp barrier
    int64_t base = 0;
    for (int i = 0; i < t; i++)
      base += rs.sums[i];
    for (int64_t i = lo; i < hi; i++) {
      rs.offsets[i] = base;
      rs.index[i] = rs.neighs + base;
      base += rs.pairs[i].first;
    }
    if (t == P - 1) {
      rs.offsets[n] = base;
      rs.index[n] = rs.neighs + base;
    }
  }
  #pragma omp parallel for schedule(dynamic, 64)
  for (NodeID u = 0; u < n; u++) {
    const NodeID nu = rs.new_ids[u];
    NodeID *out = rs.index[nu];
    for (NodeID v : g.out_neigh(u))
      *out++ = rs.new_ids[v];
    std::sort(rs.index[nu], rs.index[nu + 1]);
  }
  t.Stop();
  PrintTime("Relabel", t.Seconds());
  return OrderedCountLists(n, rs.index);
}

size_t Hybrid(const Graph &g) {
  if (WorthRelabelling(g))
    return kRelabelGap ? OrderedCount(Builder::RelabelByDegree(g)) : OrderedCountRelabelled(g);
  else
    return OrderedCount(g);
}


void PrintTriangleStats(const Graph &g, size_t total_triangles) {
  cout << total_triangles << " triangles" << endl;
}


// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(const Graph &g, size_t test_total) {
  size_t total = 0;
  vector<NodeID> intersection;
  intersection.reserve(g.num_nodes());
  for (NodeID u : g.vertices()) {
    for (NodeID v : g.out_neigh(u)) {
      auto new_end = set_intersection(g.out_neigh(u).begin(),
                                      g.out_neigh(u).end(),
                                      g.out_neigh(v).begin(),
                                      g.out_neigh(v).end(),
                                      intersection.begin());
      intersection.resize(new_end - intersection.begin());
      total += intersection.size();
    }
  }
  total = total / 6;  // each triangle was counted 6 times
  if (total != test_total)
    cout << total << " != " << test_total << endl;
  return total == test_total;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "triangle count");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  if (g.directed()) {
    cout << "Input graph is directed but tc requires undirected" << endl;
    return -2;
  }
  BenchmarkKernel(cli, g, Hybrid, PrintTriangleStats, TCVerifier);
  return 0;
}
