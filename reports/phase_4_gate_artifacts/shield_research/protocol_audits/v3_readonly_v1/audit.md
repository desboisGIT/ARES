# V3 read-only protocol audit v1

Scope: static review only; no flight runs, runtime edits or prior-evidence changes.
The exact inspected source hashes are retained in `audit.json`.

Two gaps remain in the inspected V3:

- **Timing:** map lease refresh/snapshot and guard-map updates at run_mission.py114–115 occur before control_start116. Mapper timing covers only mapper.observe. This synchronous lease/proof-map work is absent from the reported all-in shield deadline.
- **Provenance:** hashes() omits learned_command_bridge_v1/gaussian_observer.py and adapter.py despite their use for NN observation generation and intervention reporting.

Positive confirmations:

- Reached status uses actual physical waypoint proximity and final physical speed, after physical stepping; it is not anchor/proposal progress.
- The requested BPTT checkpoint is SHA-checked before loading and policy(obs) supplies the requested command. The audit independently verified the checkpoint file against its descriptor.
- ProofBundle retains the original51-source checks. Independent current-byte comparison found no mismatches across the51 frozen entries.
- Candidate scoring, online proof/renewal constructor/first command, acknowledgement, public energy ledger and receipt streaming are inside their named summed timing spans. Complete receipt materialization occurs after the finite worker loop.

These confirmations do not close the two findings or establish physical safety,
WCET or end-to-end timing. V3/evidence remain unchanged. A targeted V4 review will
be performed only if separately requested after its freeze.
