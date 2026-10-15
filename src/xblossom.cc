// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <omp.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
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
GAP Benchmark Suite (GACGE addition, not part of upstream GAP)
Kernel: X-Blossom maximum matching (XB-GAP)

Maximum-cardinality matching in a general (undirected) graph, hand-written in
GAP's style: GAP's graph, builder, command line, timer and atomics, OpenMP for
parallelism. It is the algorithm of XB-Ligra (the GACGE repository,
src/ligra/apps/XBlossomPro.C, default build: opt + triplet path table +
cycle-free blossoms, trees reused across phases), written without Ligra.

Search phases grow a forest of alternating trees from the exposed vertices;
each phase repeats, over the forest's frontier,
  1. augment -- an edge between even vertices of two different trees is an
                augmenting path: claim both trees (in tree id order, by
                compare-and-swap) and keep the edge
  2. expand  -- an even vertex's neighbor w in no tree joins it as an odd
                vertex and w's mate x as an even vertex
  3. blossom -- an edge between two even vertices of the same tree closes a
                blossom: find its base, claim the odd vertices on both chains,
                which become even after the step
until augmenting edges are found (flip their paths, next phase) or the forest
stops growing (the matching is maximum). Trees whose root is still exposed are
kept across phases. Every vertex keeps one record of three vertex ids:
  root or odd vertex                    (-1, -1, -1)
  x even by expand through v            (w, v, -1)     w = x's mate
  u even by a blossom with closing      (s, o, base)
  edge (s, o) and base
A blossom is never written out; paths are flipped by unfolding the records.

Each step spreads the frontier's edges over the threads as Ligra's sparse
edgeMap does: frontier vertices dynamically, and the edges of a vertex with
more than kSplitDegree neighbors split into chunks. Per-thread buffers collect
the new even vertices, the augmenting edges and the blossom claims; no locks.
Claims read their flag before the compare-and-swap (TryClaim).

Output (GACGE measurement format), besides GAP's Trial Time per trial:
  XBConfig: ...                                             once
  XBRound: index runtime_s matching_size valid              per trial (the check is untimed)
  XBResult: status matching_size valid rounds mean_runtime_s
*/


using namespace std;

namespace xbgap {

const int64_t kSplitDegree = 1000;  // Ligra's granular_for threshold in sparse edgeMap
const int64_t kSplitGrain = 1024;   // edges per chunk of a split vertex
const int64_t kParallelMin = 64;    // frontier vertices below which a step runs on one thread

// Vectors whose new elements are left uninitialized (they are written in
// parallel right after): std::vector zero-fills them serially on resize.
template <class T>
struct NoInitAlloc : std::allocator<T> {
  template <class U>
  struct rebind {
    using other = NoInitAlloc<U>;
  };
  NoInitAlloc() = default;
  template <class U>
  NoInitAlloc(const NoInitAlloc<U> &) noexcept {}
  template <class U>
  void construct(U *p) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void *>(p)) U;
  }
  template <class U, class... A>
  void construct(U *p, A &&...a) {
    ::new (static_cast<void *>(p)) U(std::forward<A>(a)...);
  }
};
typedef vector<NodeID, NoInitAlloc<NodeID>> Frontier;

// GACGE_XB_PHASES=1: seconds spent in each part of a matching (XBPhases line
// per trial); off by default.
const bool kPhases = getenv("GACGE_XB_PHASES") != nullptr;
enum Part { kReset, kAugment, kExpand, kBlossom, kApply, kFlip, kParts };
const char *kPartNames[kParts] = {"reset", "augment", "expand", "blossom", "apply", "flip"};
double part_seconds[kParts];
int64_t part_calls[kParts];
struct PartTimer {
  Part p;
  double t0;
  explicit PartTimer(Part p_) : p(p_), t0(kPhases ? omp_get_wtime() : 0) {}
  ~PartTimer() {
    if (kPhases) { part_seconds[p] += omp_get_wtime() - t0; part_calls[p]++; }
  }
};

struct State {
  int64_t n;
  int nthreads;
  pvector<NodeID> M;        // mate, -1 = exposed
  pvector<NodeID> is_even;  // 1 = even vertex of a tree
  pvector<NodeID> belongs;  // tree (root) of a vertex, -1 = none
  pvector<NodeID> rec;      // three vertex ids per vertex
  pvector<int> select_tree, select_match, select_blossom;  // claim flags
  vector<vector<NodeID>> out;        // per thread: new even vertices
  vector<vector<NodeID>> aug_pairs;  // per thread: augmenting edges (v, w)
  vector<vector<NodeID>> tasks;      // per thread: blossom claims (u, s, o, base)
  vector<vector<NodeID>> roots, kept_even, heavy;  // per thread: phase start, split vertices
  vector<pair<NodeID, int64_t>> chunks;            // a step's split-vertex edge chunks (v, first edge)
  Frontier v1, v2, frontier, blossom_frontier, tk, ap;  // reused across steps and phases

  explicit State(int64_t n_) : n(n_), nthreads(omp_get_max_threads()), M(n_), is_even(n_),
      belongs(n_), rec(3 * n_), select_tree(n_), select_match(n_), select_blossom(n_),
      out(nthreads), aug_pairs(nthreads), tasks(nthreads), roots(nthreads), kept_even(nthreads),
      heavy(nthreads) {
    #pragma omp parallel for
    for (int64_t x = 0; x < n_; x++) {
      M[x] = -1;
      is_even[x] = 0;
      belongs[x] = -1;
      select_tree[x] = select_match[x] = select_blossom[x] = 0;
      rec[3 * x] = rec[3 * x + 1] = rec[3 * x + 2] = -1;
    }
  }

  bool have_paths() const {
    for (auto &p : aug_pairs)
      if (!p.empty()) return true;
    return false;
  }
};

// A claim flag is read before its compare-and-swap, and a taken flag skips the
// CAS (test-and-test-and-set): the outcome of every claim is the same, but the
// failed CASes of many threads on one taken flag (a hub's tree in augment) no
// longer bounce its cache line between the cores. GACGE_XB_CAS_ONLY=1: CAS only.
const bool kCasOnly = getenv("GACGE_XB_CAS_ONLY") != nullptr;
inline bool TryClaim(int &flag) {
  if (!kCasOnly && __atomic_load_n(&flag, __ATOMIC_RELAXED) != 0) return false;
  return compare_and_swap(flag, 0, 1);
}
inline void Release(int &flag) { __atomic_store_n(&flag, 0, __ATOMIC_SEQ_CST); }

inline void SetRec(State &S, NodeID x, NodeID a, NodeID b, NodeID c) {
  NodeID *r = &S.rec[3 * static_cast<int64_t>(x)];
  r[0] = a;
  r[1] = b;
  r[2] = c;
}
inline bool RecEmpty(const State &S, NodeID x) { return S.rec[3 * static_cast<int64_t>(x)] == -1; }
// the next even vertex towards the root: v of an expand record, base of a blossom record
inline NodeID Anchor(const State &S, NodeID x) {
  const NodeID *r = &S.rec[3 * static_cast<int64_t>(x)];
  return r[2] == -1 ? r[1] : r[2];
}

// The base of the blossom closed by (v, w): the first anchor the two anchor
// chains share (each walks to its root, then switches to the other side's
// start). -1 if the chains never meet (a forest read mid-update).
NodeID FindBaseByAnchor(const State &S, NodeID v, NodeID w) {
  NodeID cv = v, cw = w;
  int64_t it = 0;
  while (cv != cw) {
    cv = RecEmpty(S, cv) ? w : Anchor(S, cv);
    cw = RecEmpty(S, cw) ? v : Anchor(S, cw);
    if (++it > 2 * S.n + 2) return -1;
  }
  return cv;
}

// Claim every vertex on s's anchor chain down to base that was odd when it
// was recorded (the w of each expand record) and is still odd: one task
// (u, s, o, base) per claim, applied after the step.
void ClaimOddOnChain(State &S, NodeID s, NodeID o, NodeID base) {
  vector<NodeID> &tk = S.tasks[omp_get_thread_num()];
  NodeID cur = s;
  while (cur != base) {
    const NodeID *r = &S.rec[3 * static_cast<int64_t>(cur)];
    if (r[0] == -1) return;  // the root without meeting the base
    if (r[2] == -1) {
      const NodeID u = r[0];
      if (!S.is_even[u] && TryClaim(S.select_blossom[u])) {
        tk.push_back(u);
        tk.push_back(s);
        tk.push_back(o);
        tk.push_back(base);
      }
    }
    cur = r[2] == -1 ? r[1] : r[2];
  }
}

// Flip the matched pairs of route(x) up to its root. A blossom record
// (s, o, base) met at u stands for (s, o), route(s) before u, and route(o),
// which reaches base and goes on as the rest of this walk: route(s) is walked
// first, route(o) kept on a stack (one entry per nesting level).
void FlipRoute(State &S, NodeID x) {
  struct Walk {
    NodeID x, stop;  // stop >= 0: end at the expand record whose w is stop (an odd vertex); -1: the root
  };
  Walk stack[256];
  int depth = 0;
  Walk cur{x, -1};
  while (true) {
    while (true) {
      const NodeID *r = &S.rec[3 * static_cast<int64_t>(cur.x)];
      if (r[0] == -1) break;  // the root
      if (r[2] == -1) {
        if (cur.stop >= 0 && r[0] == cur.stop) break;
        S.M[r[0]] = r[1];
        S.M[r[1]] = r[0];
        cur.x = r[1];
      } else {
        if (depth == 256) {
          fprintf(stderr, "XBError: blossom nesting deeper than 256 while flipping a path\n");
          abort();
        }
        stack[depth++] = Walk{r[1], cur.stop};
        S.M[r[0]] = r[1];
        S.M[r[1]] = r[0];
        cur = Walk{r[0], cur.x};
      }
    }
    if (depth == 0) break;
    cur = stack[--depth];
  }
}

// 1. Augmenting edges.
struct Augment {
  static constexpr const char *kPhase = "augment";
  State &S;
  bool cond(NodeID w) const { return S.is_even[w]; }
  void update(NodeID v, NodeID w) {
    const NodeID tv = S.belongs[v], tw = S.belongs[w];
    if (tv == tw || tv == -1 || tw == -1) return;
    // Claim the two trees in id order: whenever a pass has cross-tree edges
    // some thread gets both trees, so no pass ends empty by mutual back-off.
    const NodeID lo = min(tv, tw), hi = max(tv, tw);
    if (!TryClaim(S.select_tree[lo])) return;
    if (!TryClaim(S.select_tree[hi])) {
      Release(S.select_tree[lo]);
      return;
    }
    vector<NodeID> &ap = S.aug_pairs[omp_get_thread_num()];
    ap.push_back(v);
    ap.push_back(w);
  }
};

// 2. Expansion: w (in no tree) becomes odd, its mate x even.
struct Expand {
  static constexpr const char *kPhase = "expand";
  State &S;
  bool cond(NodeID w) const { return S.belongs[w] == -1; }
  void update(NodeID v, NodeID w) {
    const NodeID x = S.M[w];
    if (!TryClaim(S.select_match[min(w, x)])) return;
    SetRec(S, x, w, v, -1);
    S.is_even[w] = 0;
    S.is_even[x] = 1;
    S.belongs[w] = S.belongs[v];
    S.belongs[x] = S.belongs[v];
    S.out[omp_get_thread_num()].push_back(x);
  }
};

// 3. Blossoms: a closing edge claims the odd vertices on both chains.
struct Blossom {
  static constexpr const char *kPhase = "blossom";
  State &S;
  bool cond(NodeID w) const { return S.is_even[w]; }
  void update(NodeID v, NodeID w) {
    if (S.belongs[w] != S.belongs[v] || w == S.M[v] || S.belongs[w] == -1) return;
    const NodeID base = FindBaseByAnchor(S, v, w);
    if (base < 0) return;
    ClaimOddOnChain(S, v, w, base);
    ClaimOddOnChain(S, w, v, base);
  }
};

// The edges of v from position [b, e), through f.
template <typename F>
inline void ScanEdges(const NodeID *nb, int64_t b, int64_t e, NodeID v, F &f) {
  for (int64_t j = b; j < e; j++) {
    const NodeID w = nb[j];
    if (f.cond(w)) f.update(v, w);
  }
}

// One step over the frontier with f (Ligra's sparse edgeMap, no output):
// frontier vertices in dynamic chunks of 64, the edges of a vertex with more
// than kSplitDegree neighbors split into chunks of kSplitGrain edges spread
// over the threads after them. A small frontier without such a vertex runs on
// one thread (the parallel region would cost more than the step).
template <typename F>
void Step(const Graph &g, State &S, const Frontier &frontier, F f) {
  const int64_t m = frontier.size();
  if (m == 0) return;
  if (gacge::Tracing()) {  // GACGE: one XBStep line per step (gacge.h)
    int64_t fe = 0;
    #pragma omp parallel for reduction(+ : fe)
    for (int64_t i = 0; i < m; i++) fe += g.out_degree(frontier[i]);
    gacge::TraceStep(F::kPhase, "push", m, fe);
  }
  if (m < kParallelMin) {
    bool split = false;
    for (int64_t i = 0; i < m && !split; i++) split = g.out_degree(frontier[i]) > kSplitDegree;
    if (!split) {
      for (int64_t i = 0; i < m; i++) {
        const NodeID v = frontier[i];
        ScanEdges(g.out_neigh(v).begin(), 0, g.out_degree(v), v, f);
      }
      return;
    }
  }
  #pragma omp parallel
  {
    vector<NodeID> &hv = S.heavy[omp_get_thread_num()];
    #pragma omp for schedule(dynamic, 64)
    for (int64_t i = 0; i < m; i++) {
      const NodeID v = frontier[i];
      const int64_t d = g.out_degree(v);
      if (d > kSplitDegree) hv.push_back(v);
      else ScanEdges(g.out_neigh(v).begin(), 0, d, v, f);
    }
    #pragma omp single
    {
      S.chunks.clear();
      for (auto &h : S.heavy) {
        for (NodeID v : h)
          for (int64_t j = 0; j < g.out_degree(v); j += kSplitGrain) S.chunks.emplace_back(v, j);
        h.clear();
      }
    }
    const int64_t nc = S.chunks.size();
    #pragma omp for schedule(dynamic, 1)
    for (int64_t c = 0; c < nc; c++) {
      const NodeID v = S.chunks[c].first;
      const int64_t b = S.chunks[c].second, e = min(b + kSplitGrain, g.out_degree(v));
      ScanEdges(g.out_neigh(v).begin(), b, e, v, f);
    }
  }
}

// dst[0, n) = src[0, n), in parallel when large.
void ParCopy(const NodeID *src, int64_t n, NodeID *dst) {
  if (n <= (1 << 16)) {
    copy(src, src + n, dst);
    return;
  }
  #pragma omp parallel for
  for (int64_t i = 0; i < n; i++) dst[i] = src[i];
}

// dst = a followed by b.
void Concat(Frontier &dst, const Frontier &a, const Frontier &b) {
  dst.resize(a.size() + b.size());
  ParCopy(a.data(), a.size(), dst.data());
  ParCopy(b.data(), b.size(), dst.data() + a.size());
}

// The per-thread buffers bufs, in thread order, into v (one parallel copy per
// thread when large).
void Gather(vector<vector<NodeID>> &bufs, Frontier &v) {
  const int64_t W = bufs.size();
  vector<int64_t> off(W + 1, 0);
  for (int64_t t = 0; t < W; t++) off[t + 1] = off[t] + bufs[t].size();
  v.resize(off[W]);
  #pragma omp parallel for schedule(dynamic, 1) if (off[W] > (1 << 16))
  for (int64_t t = 0; t < W; t++) {
    copy(bufs[t].begin(), bufs[t].end(), v.begin() + off[t]);
    bufs[t].clear();
  }
}

// One search phase; fills S.aug_pairs or leaves them empty when the forest is exhausted.
void SearchPhase(const Graph &g, State &S) {
  const int64_t n = S.n;
  vector<vector<NodeID>> &roots = S.roots, &kept_even = S.kept_even;
  PartTimer *reset = new PartTimer(kReset);
  // Reuse trees whose root is still exposed. A static schedule hands each
  // thread one contiguous range, so the gathered frontiers are in id order.
  #pragma omp parallel
  {
    vector<NodeID> &r = roots[omp_get_thread_num()], &k = kept_even[omp_get_thread_num()];
    #pragma omp for schedule(static)
    for (int64_t x = 0; x < n; x++) {
      S.select_tree[x] = S.select_match[x] = S.select_blossom[x] = 0;
      const NodeID root = S.belongs[x];
      if (S.M[x] == -1) {
        // exposed vertex: the root of a new tree
        S.is_even[x] = 1;
        S.belongs[x] = x;
        SetRec(S, x, -1, -1, -1);
        r.push_back(x);
      } else if (root == -1 || S.M[root] != -1) {
        // in no tree, or in a tree whose root got matched
        S.is_even[x] = 0;
        S.belongs[x] = -1;
        SetRec(S, x, -1, -1, -1);
      } else if (S.is_even[x]) {
        k.push_back(x);  // in a tree whose root is still exposed: kept
      }
    }
  }
  Frontier &v1 = S.v1, &v2 = S.v2, &frontier = S.frontier, &blossom_frontier = S.blossom_frontier;
  Gather(roots, v1);
  Gather(kept_even, v2);
  Concat(frontier, v1, v2);
  delete reset;
  while (true) {
    { PartTimer t(kAugment); Step(g, S, frontier, Augment{S}); }
    if (S.have_paths()) return;
    // expand from every even vertex of the frontier: v1 = the new even vertices
    { PartTimer t(kExpand); Step(g, S, frontier, Expand{S}); Gather(S.out, v1); }
    // blossoms from the new even vertices and the previous blossom step's
    { PartTimer t(kBlossom); Concat(blossom_frontier, v1, v2); Step(g, S, blossom_frontier, Blossom{S}); }
    // the claimed odd vertices become even around their blossoms
    PartTimer apply(kApply);
    Frontier &tk = S.tk;
    Gather(S.tasks, tk);
    const int64_t nt = tk.size() / 4;
    v2.resize(nt);
    #pragma omp parallel for if (nt > (1 << 12))
    for (int64_t i = 0; i < nt; i++) {
      const NodeID u = tk[4 * i];
      SetRec(S, u, tk[4 * i + 1], tk[4 * i + 2], tk[4 * i + 3]);
      S.is_even[u] = 1;
      v2[i] = u;
    }
    Concat(frontier, v1, v2);
    if (frontier.empty()) return;
  }
}

pvector<NodeID> MaximumMatching(const Graph &g) {
  State S(g.num_nodes());
  while (true) {
    SearchPhase(g, S);
    if (!S.have_paths()) break;
    // each augmenting edge (v, w) owns its two trees: the flips do not overlap
    PartTimer t(kFlip);
    Frontier &ap = S.ap;
    Gather(S.aug_pairs, ap);
    const int64_t np = ap.size() / 2;
    #pragma omp parallel for schedule(dynamic, 16) if (np > 64)
    for (int64_t i = 0; i < np; i++) {
      const NodeID v = ap[2 * i], w = ap[2 * i + 1];
      S.M[v] = w;
      S.M[w] = v;
      FlipRoute(S, v);
      FlipRoute(S, w);
    }
  }
  return move(S.M);
}

// Matching size if mate is a valid matching of g (mutual, along edges), else -1.
int64_t CheckMatching(const Graph &g, const pvector<NodeID> &M) {
  int64_t matched = 0, bad = 0;
  #pragma omp parallel for reduction(+ : matched, bad)
  for (NodeID v = 0; v < g.num_nodes(); v++) {
    const NodeID u = M[v];
    if (u == -1) continue;
    if (u < 0 || u >= g.num_nodes() || M[u] != v) { bad++; continue; }
    bool edge = false;
    for (NodeID x : g.out_neigh(v))
      if (x == u) { edge = true; break; }
    if (edge) matched++;
    else bad++;
  }
  return bad ? -1 : matched / 2;
}

}  // namespace xbgap


class CLXBlossom : public CLApp {
  std::string dataset_ = "unknown";

 public:
  CLXBlossom(int argc, char** argv, std::string name) : CLApp(argc, argv, name) {
    get_args_ += "d:";
    AddHelpLine('d', "name", "dataset name for the XBConfig line", "unknown");
  }

  void HandleArg(signed char opt, char* opt_arg) override {
    switch (opt) {
      case 'd': dataset_ = std::string(opt_arg); break;
      default: CLApp::HandleArg(opt, opt_arg);
    }
  }

  std::string dataset() const { return dataset_; }
};


int main(int argc, char* argv[]) {
  CLXBlossom cli(argc, argv, "x-blossom maximum matching");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  if (g.directed()) {
    cout << "Input graph is directed but x-blossom requires undirected" << endl;
    return -2;
  }
  g.PrintStats();
  int64_t max_degree = 0;
  #pragma omp parallel for reduction(max : max_degree)
  for (NodeID v = 0; v < g.num_nodes(); v++) max_degree = max(max_degree, g.out_degree(v));
  printf("XBConfig: arm=gap variant=xb-gap reuse=1 triplet=1 lb=split-%" PRId64 " dataset=%s nodes=%" PRId64
         " edges=%" PRId64 " max_degree=%" PRId64 " trials=%d threads=%d\n",
         xbgap::kSplitDegree, cli.dataset().c_str(), g.num_nodes(), g.num_edges(), max_degree,
         cli.num_trials(), omp_get_max_threads());
  // GAP's BenchmarkKernel loop, plus the untimed matching check of each trial
  Timer trial_timer;
  double total_seconds = 0;
  int64_t first_size = -2;
  bool all_valid = true, all_same = true;
  for (int iter = 0; iter < cli.num_trials(); iter++) {
    fill(xbgap::part_seconds, xbgap::part_seconds + xbgap::kParts, 0.0);
    fill(xbgap::part_calls, xbgap::part_calls + xbgap::kParts, 0);
    gacge::TrialBegin(iter);  // GACGE: perf region / traversal trace (gacge.h)
    trial_timer.Start();
    pvector<NodeID> mate = xbgap::MaximumMatching(g);
    trial_timer.Stop();
    gacge::TrialEnd(iter);
    if (gacge::Tracing()) gacge::TraceEnd(g.num_nodes(), g.num_edges_directed());
    PrintTime("Trial Time", trial_timer.Seconds());
    total_seconds += trial_timer.Seconds();
    const int64_t size = xbgap::CheckMatching(g, mate);
    const bool valid = size >= 0;
    all_valid = all_valid && valid;
    if (first_size == -2) first_size = size;
    else if (size != first_size) all_same = false;
    printf("XBRound: index=%d runtime_s=%.9f matching_size=%" PRId64 " valid=%d\n", iter,
           trial_timer.Seconds(), size, valid ? 1 : 0);
    if (xbgap::kPhases) {
      printf("XBPhases: index=%d", iter);
      for (int p = 0; p < xbgap::kParts; p++)
        printf(" %s_s=%.6f %s_calls=%" PRId64, xbgap::kPartNames[p], xbgap::part_seconds[p],
               xbgap::kPartNames[p], xbgap::part_calls[p]);
      printf("\n");
    }
    fflush(stdout);
  }
  PrintTime("Average Time", total_seconds / cli.num_trials());
  const char *status = !all_valid ? "invalid_matching" : (!all_same ? "matching_size_varies" : "ok");
  printf("XBResult: status=%s matching_size=%" PRId64 " valid=%d rounds=%d mean_runtime_s=%.9f\n", status,
         first_size, all_valid ? 1 : 0, cli.num_trials(), total_seconds / cli.num_trials());
  return all_valid ? 0 : 1;
}
