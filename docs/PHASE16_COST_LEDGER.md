# Phase 16 spending ledger

Authorized new compute: $2 USD total, $1/session, 30 minutes/session, one existing
`forge-gpu` NVIDIA L4. Previous phase budgets do not carry forward.

Preflight: authenticated provider API reported all visible Studios with
`requested=null`, `inUse=null`; on-demand L4 $1.68/hour; sufficient credits
(`REQUIRED_BALANCE_REASON_SUFFICIENT_CREDITS`). Exact account balance and invoices
are not exposed by this endpoint. Auto-sleep enabled, 900 seconds.

| Session | Reservation/start UTC | Conservative independent stop verification UTC | Minutes through verification | Rate USD/hour | Reservation | Estimate rounded up |
|---|---|---|---:|---:|---:|---:|
| 1 | 2026-10-09T05:06:15.8655795Z | 2026-10-09T05:31:00.6381211Z | 24.75 | 1.68 | $0.84 | $0.70 |

Actual invoice unavailable. Conservative estimated spend **$0.70**; remaining
estimated Phase 16 budget **$1.30**. No second session was used.

Provider-side planned runtime was 28 minutes. Independent local watchdog PID 15080
was armed before provisioning; the remote watchdog reported readiness with absolute
UTC deadline 2026-10-09T05:34:15.8655795Z. Provider MaxRuntime alone was not counted
as reliable protection. Explicit stop reported `stopped (instance unreachable)`;
independent API inspection returned `requested=null`, `inUse=null` for all visible
Studios. The local watchdog was cancelled only after this independent verification.

Evidence: `benchmarks/results/phase16/cost-session1.json`, `preflight.txt`,
`watchdog.log`, `stop.txt` and `final-cloud-state.txt`. Estimates cover provisioning
and all time through conservative stop verification, rather than only benchmark
runtime. No prior-phase budget was used.
