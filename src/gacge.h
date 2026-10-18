// GACGE measurement hooks (not part of upstream GAP). Every hook is off unless
// its environment variable is set, so a timing run executes none of them
// beyond a load of a static flag per trial or per step.
//
// perf region (GACGE_PERF_CTL, GACGE_PERF_ACK: the control and ack FIFOs of
// `perf stat -D -1 --control fifo:<ctl>,<ack>`, see measure/common/perf_region.sh):
// the counters run only inside the trials, from just before a trial's kernel to
// just after it, so reading, building and converting the graph and verifying
// the result are not counted. GACGE_PERF_SKIP=<k> leaves the first k trials
// uncounted (warm-up; default 0). GACGE_PERF_PHASE=<name> counts only inside
// that phase of the counted trials (PerfPhase regions: X-Blossom's augment,
// expand, blossom; BFS's td, bu) instead of the whole kernel.
//
// traversal trace (GACGE_TRAVERSAL_TRACE=<trial>): one XBStep line per step of
// that trial (0-based) and an XBTraversal line after it, in the format of the
// former Ligra trace (measure/<algo>-cpu/targets/trace):
//   XBStep: run=0 step=<i> phase=<p> mode=<push|pull> frontier_v=<n> frontier_e=<m>
//   XBTraversal: run=0 steps=<s> frontier_v=<sum> frontier_e=<sum> nodes=<n> edges=<m>

#ifndef GACGE_H_
#define GACGE_H_

#include <fcntl.h>
#include <unistd.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gacge {

inline int &trial() {
  static int t = -1;
  return t;
}

// ---- perf region ------------------------------------------------------------
struct PerfCtl {
  int ctl = -1, ack = -1, skip = 0;
  PerfCtl() {
    const char *c = getenv("GACGE_PERF_CTL"), *a = getenv("GACGE_PERF_ACK");
    if (c == nullptr || a == nullptr) return;
    ctl = open(c, O_WRONLY | O_CLOEXEC);
    ack = open(a, O_RDONLY | O_CLOEXEC);
    if (ctl < 0 || ack < 0) {
      perror("gacge: perf control FIFO");
      exit(3);
    }
    const char *s = getenv("GACGE_PERF_SKIP");
    skip = s ? atoi(s) : 0;
  }
  void Send(const char *cmd) {
    if (write(ctl, cmd, strlen(cmd)) < 0) { perror("gacge: perf control"); exit(3); }
    char buf[16];
    if (read(ack, buf, sizeof buf) <= 0) { perror("gacge: perf ack"); exit(3); }
  }
};

inline PerfCtl &perf() {
  static PerfCtl p;
  return p;
}

inline const char *PerfPhase() {
  static const char *ph = getenv("GACGE_PERF_PHASE");
  return ph;
}

// Around one trial's kernel (the region GAP times); with GACGE_PERF_PHASE the
// counters run only inside that phase's PerfPhase regions instead.
inline void TrialBegin(int t) {
  trial() = t;
  PerfCtl &p = perf();
  if (p.ctl >= 0 && t >= p.skip && PerfPhase() == nullptr) p.Send("enable\n");
}
inline void TrialEnd(int t) {
  PerfCtl &p = perf();
  if (p.ctl >= 0 && t >= p.skip && PerfPhase() == nullptr) p.Send("disable\n");
}

// A scope of one phase of a trial: counted when GACGE_PERF_PHASE names it.
struct PerfPhaseRegion {
  bool on = false;
  explicit PerfPhaseRegion(const char *name) {
    const char *ph = PerfPhase();
    PerfCtl &p = perf();
    if (ph != nullptr && p.ctl >= 0 && trial() >= p.skip && strcmp(ph, name) == 0) {
      p.Send("enable\n");
      on = true;
    }
  }
  ~PerfPhaseRegion() {
    if (on) perf().Send("disable\n");
  }
};

// ---- traversal trace ------------------------------------------------------------
inline int TraceTrial() {
  static const int t = [] {
    const char *e = getenv("GACGE_TRAVERSAL_TRACE");
    return e ? atoi(e) : -1;
  }();
  return t;
}
inline bool Tracing() { return TraceTrial() >= 0 && trial() == TraceTrial(); }

struct TraceTotals {
  int64_t steps = 0, v = 0, e = 0;
  int64_t work_e = 0;  // edges a kernel reports as processed (TraceWork), e.g. sssp's relaxations
};
inline TraceTotals &totals() {
  static TraceTotals t;
  return t;
}

inline void TraceStep(const char *phase, const char *mode, int64_t frontier_v, int64_t frontier_e) {
  TraceTotals &t = totals();
  printf("XBStep: run=0 step=%" PRId64 " phase=%s mode=%s frontier_v=%" PRId64 " frontier_e=%" PRId64 "\n",
         t.steps, phase, mode, frontier_v, frontier_e);
  t.steps++;
  t.v += frontier_v;
  t.e += frontier_e;
}

// Edges processed outside the traced steps' frontiers (sssp: every relaxation,
// the thread-local bin fusion included); summed into XBTraversal's work_e.
inline void TraceWork(int64_t edges) { __atomic_fetch_add(&totals().work_e, edges, __ATOMIC_RELAXED); }

inline void TraceEnd(int64_t nodes, int64_t edges) {
  TraceTotals &t = totals();
  printf("XBTraversal: run=0 steps=%" PRId64 " frontier_v=%" PRId64 " frontier_e=%" PRId64 " work_e=%" PRId64
         " nodes=%" PRId64 " edges=%" PRId64 "\n", t.steps, t.v, t.e, t.work_e, nodes, edges);
  fflush(stdout);
  t = TraceTotals();
}

// ---- sync stats -----------------------------------------------------------------
// GACGE_SYNC_STATS=1 (compiled only with -DGACGE_SYNC_STATS_BUILD=1; the timed
// builds have none of it): for each worksharing loop a kernel marks, how long the
// threads wait at its barrier for the slowest one. GACGE_SYNC_BEGIN(r) as a thread
// enters the loop, GACGE_SYNC_END(r, name) once it has done its share (the loop
// `omp for ... GACGE_NOWAIT`, so it does not wait there): each thread's busy time is
// from its BEGIN to its END, the loop's span from the first BEGIN to the last END,
// then the barrier. Summed per region r (0-7) over a trial; after each counted
// trial (GACGE_SYNC_SKIP=<k>, default 1: the warm-up left out) one line per region:
//   XBSync: trial=<t> region=<name> loops=<k> threads=<P> span_s=<sum of spans>
//           busy_s=<sum of busy> wait_s=<sum of waits at the barrier> skew_s=<sum of late
//           entries> idle_s=<one-thread steps: the others' time> util=<busy / (P x span)>
// (P x span = busy + wait + skew: a thread's busy time, its wait at the barrier for the
// slowest, and how much later than the first it entered the loop; + idle for the
// GACGE_SYNC_SERIAL steps.)
// GACGE_SYNC_SERIAL(r, name, s) adds s seconds of work on one thread while the others
// idle (busy s, span s), e.g. XB-GAP's small frontiers.
#ifndef GACGE_SYNC_STATS_BUILD
#define GACGE_SYNC_STATS_BUILD 0
#endif
}  // namespace gacge
#if GACGE_SYNC_STATS_BUILD
#include <omp.h>
#include <algorithm>
namespace gacge {
namespace sync {
constexpr int kRegions = 8, kThreads = 512;
struct alignas(64) Slot { double begin = 0, end = 0; };
struct Acc { const char *name = nullptr; int64_t loops = 0; int threads = 0; double span = 0, busy = 0, wait = 0, skew = 0, idle = 0; };
inline Slot (&slots())[kRegions][kThreads] { static Slot s[kRegions][kThreads]; return s; }
inline Acc (&acc())[kRegions] { static Acc a[kRegions]; return a; }
inline bool On() {
  static const bool on = getenv("GACGE_SYNC_STATS") != nullptr;
  return on;
}
inline bool Counted() {
  static const int skip = getenv("GACGE_SYNC_SKIP") ? atoi(getenv("GACGE_SYNC_SKIP")) : 1;
  return On() && trial() >= skip;
}
inline void Begin(int r) { if (Counted()) slots()[r][omp_get_thread_num()].begin = omp_get_wtime(); }
inline void End(int r) { if (Counted()) slots()[r][omp_get_thread_num()].end = omp_get_wtime(); }
// one thread, after the barrier
inline void Fold(int r, const char *name) {
  if (!Counted()) return;
  const int P = omp_get_num_threads();
  double b = slots()[r][0].begin, e = slots()[r][0].end, busy = 0, wait = 0, skew = 0;
  for (int t = 0; t < P; t++) {
    b = std::min(b, slots()[r][t].begin);
    e = std::max(e, slots()[r][t].end);
  }
  for (int t = 0; t < P; t++) {  // span = busy + wait (at the barrier) + skew (entering late), per thread
    busy += slots()[r][t].end - slots()[r][t].begin;
    wait += e - slots()[r][t].end;
    skew += slots()[r][t].begin - b;
  }
  Acc &a = acc()[r];
  a.name = name;
  a.loops++;
  a.threads = std::max(a.threads, P);
  a.span += e - b;
  a.busy += busy;
  a.wait += wait;
  a.skew += skew;
}
inline void Serial(int r, const char *name, double s) {
  if (!Counted()) return;
  Acc &a = acc()[r];
  a.name = name;
  a.loops++;
  a.threads = std::max(a.threads, omp_get_max_threads());
  a.span += s;
  a.busy += s;
  a.idle += (omp_get_max_threads() - 1) * s;  // the other threads, with nothing to do
}
inline void Report() {
  if (!Counted()) return;
  for (Acc &a : acc()) {
    if (a.loops)
      printf("XBSync: trial=%d region=%s loops=%" PRId64 " threads=%d span_s=%.9f busy_s=%.9f wait_s=%.9f"
             " skew_s=%.9f idle_s=%.9f util=%.4f\n", trial(), a.name, a.loops, a.threads, a.span, a.busy, a.wait, a.skew, a.idle,
             a.span > 0 ? a.busy / (a.threads * a.span) : 0.0);
    a = Acc();
  }
  fflush(stdout);
}
}  // namespace sync
}  // namespace gacge
#define GACGE_NOWAIT nowait
#define GACGE_SYNC_BEGIN(r) gacge::sync::Begin(r)
#define GACGE_SYNC_END(r, name)    \
  do {                             \
    gacge::sync::End(r);           \
    _Pragma("omp barrier")         \
    _Pragma("omp single")          \
    gacge::sync::Fold(r, name);    \
  } while (0)
#define GACGE_SYNC_SERIAL(r, name, s) gacge::sync::Serial(r, name, s)
#define GACGE_SYNC_REPORT() gacge::sync::Report()
#else
#define GACGE_NOWAIT
#define GACGE_SYNC_BEGIN(r) ((void)0)
#define GACGE_SYNC_END(r, name) ((void)0)
#define GACGE_SYNC_SERIAL(r, name, s) ((void)0)
#define GACGE_SYNC_REPORT() ((void)0)
#endif

#endif  // GACGE_H_
