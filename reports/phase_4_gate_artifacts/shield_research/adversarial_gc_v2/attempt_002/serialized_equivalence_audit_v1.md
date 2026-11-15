# Attempt002: read-only serialization audit v1

The unchanged original criteria are satisfied by the retained physical/timing,
provenance and canonical serialized data. No flight was rerun and no original
wrapper flag or report was rewritten.

The wrapper's false equivalence/overall flags came from one reporting defect:
it compared old JSON-decoded **list** anchors with new equal-valued in-memory
**tuple** anchors. Python reports tuple!=list. Every one of the1,000 rows per
scenario was falsely flagged solely on its anchor container type.

The separate audit reloads both saved JSON reports, removes only `guard_timing`,
and compares canonical serialized JSON for every remaining field. All three
1,000-row traces match exactly, with matching SHA256 digests and zero differing
Float32 command words. Root independently confirmed the same zero differences.
`serialized_equivalence_audit_v1.json` retains digests, comparison details and the
original false flags; `derived_summary_v1.json` records the derived verdict.

Physical results remain the exact old scenarios: zero actual-C/queue/contact/
unknown failures,686 anchor accepts and314 cached fallbacks per scenario, with
139 geometry refusals while candidate C still fits. Timing maxima were3.832ms
(unknown plane),3.598ms(occupied plane),3.460ms(corner), with zero calls over10ms.
No cyclic collection occurred in any bounded1,000-step context; prior GC-enabled
states and parent settings were restored. Explicit preflight collection cost
36.10/35.49/38.05ms respectively and remains recorded outside control timing.

Complete outer source inventories stayed stable. Original source/evidence hashes
are unchanged. The exact one-file unused-launcher compatibility binding passed,
with the launcher absent before/after, no observed imports, and no shared-code
import/lookup references. The child-only output-root rebinding was restored;
original import/parent roots and function bytecode were unchanged.

Attempt001 remains a zero-step preflight refusal. All original timing failures
remain retained. In particular, this result does not establish a cause for the
older corner outlier, whose GC deltas were zero. These are finite conditional
SIL observations, not hardware certification, WCET or a universal10ms guarantee.
