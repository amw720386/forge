# Phase 13 cost ledger

Authorization: $6 new estimated spend, $2/session, at most 45 minutes,
one NVIDIA L4 in existing `forge-gpu`. No previous-phase credit rollover.

Preflight: authentication succeeded; Studio `requested=null`, `inUse=null`.
AWS accelerator catalog on 2026-10-08: `lit-l4-1`, `g6.4xlarge`,
`lightning-public-prod`, $1.68/hour on demand ($1.01904/hour spot).
Provider `required-balance-status` reported sufficient credits; no numerical
balance was exposed. Reservation $1.20, including shutdown margin.

| Session | Provisioning start UTC | Stop verified by UTC | Duration upper bound | Estimated spend | State |
|---|---|---|---|---|---|
| 1 | 2026-10-08 17:54:17 | 2026-10-08 18:31:00 | 36m 43s | $1.03 | stopped |

Estimate conservatively charges the whole provisioning/shutdown interval:
36m43s × $1.68/hour = $1.0281, rounded up to $1.03. This is not an invoice;
taxes, storage, and provider billing adjustments are not exposed here.
Remaining estimated Phase 13 compute budget: **$4.97**. One session used.

Safeguards: auto-sleep enabled (900 seconds); requested SDK MaxRuntime 2400
seconds. Live provider state showed requested duration 0 and no termination
time, so MaxRuntime was not treated as enforced. Independent remote SDK
watchdog and hidden local PowerShell watchdog both used absolute 18:31 UTC
deadline, which includes provisioning. Explicit stop completed before deadline:
`before=running after=stopped (instance unreachable)`.
Subsequent authenticated provider inspection returned `requested=null`,
`inUse=null`, `collabStatus=null`; no billable instance remained.
Model/toolchain caches and content-hashed deployments were preserved.
