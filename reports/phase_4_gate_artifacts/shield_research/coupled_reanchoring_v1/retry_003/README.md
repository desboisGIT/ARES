# Frozen v2 coupled run: retry_003

All eight predeclared cases completed 1,000 macrosteps with no actual-C misses,
queue mismatches, physical collisions or unknown entries. All 8,000 calls returned
positive conditional guard decisions; 6,181 new anchors were accepted and 1,819
rejections retained the previous backup. Every trajectory and command/oracle
field present in retry_002 is exactly unchanged.

The fresh strict timing test passed: decide+acknowledge averaged 2.304 ms and
peaked at 4.271 ms, with zero calls over 10 ms. The maximum occurred at case2,
tick38. Its decide/ack times were 4.204 ms/.0667 ms, and GC collection deltas were
zero. Per-tick timings and GC deltas are retained in each trace. No observed guard
call triggered a GC collection. GC snapshots are sampled immediately outside
call timers; they are diagnostics, not a causal scheduling proof.

Every worker used one pinned CPU0–7, one logical processor per physical core,
with no exclusive-core reservation claim. All expanded executing-helper,
physical-source, bounded-input and proof-file hashes were stable and identical
across workers. The v1 and v2 point implementations and exact compatibility
artifact are all bound. The changed kernel preserves the same mathematical law
and final command-error contract; the earlier timing failure remains preserved
under retry_002 and was not rerun unchanged to discard a rare miss.

`full_sweep.json` retains all cases and trajectories; `summary.json` provides
aggregate results and its source-report hash. This is finite software SIL
functional/timing evidence for the stated hard service and verified CPU backend.
It is not a universal worst-case execution-time proof, live sensor-to-actuator
bound, hardware certificate or certification of the Gaussian runtime. Supplier,
plant and independent oracle time are excluded from guard timing. No overall
phase-pass flag is set, and `flight_certificate` remains false.
