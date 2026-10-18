// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"


/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This legacy PR implementation uses the traditional iterative approach. This is
done to ease comparisons to other implementations (often use same algorithm),
but it is not necessarily the fastest way to implement it. It performs each
iteration as a sparse-matrix vector multiply (SpMV), and values are not visible
until the next iteration (like Jacobi-style method).
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;

// GACGE: PageRankPull's scores and contributions, allocated once outside the
// trials; PreparePR sets the scores to 1/n as PageRankPull did, untimed
// (BenchmarkKernelPrepared).
struct PRState {
  pvector<ScoreT> scores, outgoing_contrib;
  explicit PRState(const Graph &g) : scores(g.num_nodes()), outgoing_contrib(g.num_nodes()) {}
};

void PreparePR(const Graph &g, PRState &s) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  #pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    s.scores[n] = init_score;
}

pvector<ScoreT>& PageRankPull(const Graph &g, PRState &s, int max_iters, double epsilon = 0,
                              bool logging_enabled = false) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> &scores = s.scores;
  pvector<ScoreT> &outgoing_contrib = s.outgoing_contrib;
  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    #pragma omp parallel for
    for (NodeID n=0; n < g.num_nodes(); n++)
      outgoing_contrib[n] = scores[n] / g.out_degree(n);
    #pragma omp parallel
    {
      GACGE_SYNC_BEGIN(0);  // GACGE: barrier wait (gacge.h sync stats)
      #pragma omp for reduction(+ : error) schedule(dynamic, 16384) GACGE_NOWAIT
      for (NodeID u=0; u < g.num_nodes(); u++) {
        ScoreT incoming_total = 0;
        for (NodeID v : g.in_neigh(u))
          incoming_total += outgoing_contrib[v];
        ScoreT old_score = scores[u];
        scores[u] = base_score + kDamp * incoming_total;
        error += fabs(scores[u] - old_score);
      }
      GACGE_SYNC_END(0, "pull");
    }
    if (logging_enabled)
      PrintStep(iter, error);
    if (error < epsilon)
      break;
  }
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incoming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incoming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incoming_sums[n] - scores[n]);
    incoming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  PRState state(g);  // GACGE: allocated once, prepared untimed per trial
  auto Prepare = [&state] (const Graph &g) { PreparePR(g, state); };
  auto PRBound = [&state, &cli] (const Graph &g) -> pvector<ScoreT>& {
    return PageRankPull(g, state, cli.max_iters(), cli.tolerance(), cli.logging_en());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernelPrepared(cli, g, Prepare, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
