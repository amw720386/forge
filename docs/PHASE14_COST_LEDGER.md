# Phase 14 cost ledger

Authorization: $6 total new estimated compute spend, $2/session, 45 minutes maximum, one existing forge-gpu NVIDIA L4. Prior budgets excluded.

Preflight authentication succeeded; provider L4 enabled, not out of capacity, on-demand $1.68/hour. Required balance sufficient.

One session completed; no GPU is running.

| Item | Session 1 |
|---|---|
| Provisioning start UTC | 2026-10-08 18:52:13.5040998 |
| Independent watchdog deadline UTC | 2026-10-08 19:34:13.5040998 |
| Verified stopped by UTC | 2026-10-08 19:33:21.0421808 |
| Conservative duration | Full provisioning-to-stop interval: 41 minutes 8 seconds |
| Estimated compute, rounded upward | **$1.16** |
| Actual invoice | Unavailable |
| Remaining Phase 14 budget | **$4.84** |

The initial $1.26 reservation was released and replaced by the final estimate.
Both session time and cost stayed below the authorized limits. No second session
is planned. This is an estimate, not an actual provider invoice.

Independent hidden local and remote watchdogs were verified during the run;
15-minute auto-sleep was configured. The SDK's 40-minute MaxRuntime request
returned provider value 0 and was not treated as an enforced guard. The local
watchdog was canceled only after shutdown verification.

`forge-cloud stop` returned `before=running after=stopped (instance unreachable)`.
Authenticated final inspection showed requested and in-use instances null for
all three visible Studios. Evidence: `benchmarks/results/phase14/shutdown-verification.json`
and `shutdown-inspection.txt`. Failed builds/conversion and an interrupted eager
run were repaired inside this session and excluded from performance claims.
Results were fetched before shutdown; documentation was finished locally.
