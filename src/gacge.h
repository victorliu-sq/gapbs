// GACGE measurement hooks (not part of upstream GAP). Every hook is off unless
// its environment variable is set, so a timing run executes none of them
// beyond a load of a static flag per trial or per step.
//
// perf region (GACGE_PERF_CTL, GACGE_PERF_ACK: the control and ack FIFOs of
// `perf stat -D -1 --control fifo:<ctl>,<ack>`, see measure/common/perf_region.sh):
// the counters run only inside the trials, from just before a trial's kernel to
// just after it, so reading, building and converting the graph and verifying
// the result are not counted. GACGE_PERF_SKIP=<k> leaves the first k trials
// uncounted (warm-up; default 0).
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

// Around one trial's kernel (the region GAP times).
inline void TrialBegin(int t) {
  trial() = t;
  PerfCtl &p = perf();
  if (p.ctl >= 0 && t >= p.skip) p.Send("enable\n");
}
inline void TrialEnd(int t) {
  PerfCtl &p = perf();
  if (p.ctl >= 0 && t >= p.skip) p.Send("disable\n");
}

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

}  // namespace gacge

#endif  // GACGE_H_
