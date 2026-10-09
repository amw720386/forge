# Phase 15 cloud cost ledger

Authorization: $6 of new Phase 15 compute, $2/session, 45 minutes/session,
one existing `forge-gpu` NVIDIA L4 at a time. No earlier-phase budget carried over.
Authenticated provider catalog and sufficient credit were checked before each
session. The on-demand L4 rate was $1.68/hour for all three starts; the CLI's
generic historical rate was overridden. Actual provider invoice is unavailable.

| Session | Provisioning start UTC | Conservative verified stop UTC | Duration through verification | Estimated USD, rounded up |
|---|---|---|---:|---:|
| 1 | 2026-10-08T20:15:30.6563281+00:00 | 2026-10-08T20:57:52.3186008Z | 42.36 min | $1.19 |
| 2 | 2026-10-09T02:48:25.708021+00:00 | 2026-10-09T03:30:08.386971+00:00 | 41.71 min | $1.17 |
| 3 | 2026-10-09T03:30:45.456427+00:00 | 2026-10-09T03:52:51.068234+00:00 | 22.09 min | $0.62 |

Conservative compute estimate: **$2.98**. Estimated budget remaining:
**$3.02**. Keep **$0.15** reserved for delayed billing; unreserved remainder
is **$2.87**. These are elapsed-time estimates, not an actual invoice.
Reservations were $1.26, $1.26, and $0.70 respectively, including shutdown margin.

Session 1's independent watchdog stopped the Studio. Sessions 2 and 3 were
explicitly stopped, with independent absolute-deadline remote and hidden local
watchdogs also armed. Session 3's deadline was 2026-10-09T03:52:45.4564278Z;
verification completed about six seconds later, well within the 45-minute limit.
The local watchdog is polled every ten seconds; stop verification also takes time.
Provider MaxRuntime was not visibly enforced, so it was never counted as protection.
The remote SDK watchdog now autodetects the current Studio, checks its exact name,
and forbids creation; this avoids treating a display organization name as an SDK slug.

Evidence: `session1-watchdog-stop.txt`, `session2-stop.txt`, `session3-stop.txt`,
`session2-preflight.txt`, `session3-preflight.txt`, `session3-watchdog.log`,
and `final-cloud-state.txt` under `benchmarks/results/phase15/`.
Final independent API inspection returned requested=null and inUse=null for all
visible Studios. Local watchdog cancellation followed verified shutdown.
