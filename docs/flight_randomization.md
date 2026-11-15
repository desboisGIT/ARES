# Episode profile randomization

`ares.flight_randomization.sample_flight_profile(profile, seed, mode="random")`
validates a v1 or v2 profile and returns a deep copy. The schema, profile ID,
units and all declared ranges are preserved. Every parameter's `nominal` holds
its episode value, including compute latency, inertia's three components and
all five material probabilities. The original profile remains unchanged.

The seed must be an integer (booleans are rejected). Each parameter component
uses its own standard-library PRNG seeded by SHA-256 of the versioned sampler
namespace, episode seed, parameter name and component index. Dictionary order
therefore does not affect samples. Floating point scalars and vector components
use independent uniform draws in their declared ranges. Depth width, height and
edge width use inclusive integer draws. Constant ranges remain constant. The
profile validator guarantees the supported cross-parameter constraints for all
points in the supplied ranges; the sampled result is validated again.

`mode="low"` and `mode="high"` select every component's exact lower or upper
endpoint, respectively. `mode="nominal"` returns an independent copy with the
input nominal values. Endpoint modes are deliberate audit probes, not statements
that either corner is the physically hardest flight case.

Use `sampled_profile_values(profile)` to record a JSON-serializable copy of every
sampled value alongside the episode seed and sampling mode. This information is
kept outside the strict profile schema; no provenance fields are inserted into
the profile and the profile ID continues to identify its source assumptions.

`randomization_manifest(profiles)` accepts one profile or a nonempty iterable of
sampled profiles sharing schema, parameter names and declared ranges. It returns
sample, parameter and scalar-dimension counts, then reports each parameter's
component ranges, observed minimum/maximum, bound checks and exact endpoint
coverage. `all_endpoints_seen` becomes true when both low and high probes are
included, including constant components. The v2 fixture has 29 parameters and 35
scalar dimensions. Invalid profiles or inconsistent ranges fail the audit rather
than being silently combined.

Sampling coverage is distinct from simulator wiring. Sampling battery energy,
optical-flow rate or compute latency does not establish that a simulator consumes
those values. Flight integration and its evidence must separately identify the
consumer for each parameter and report unsupported parameters honestly. This
module performs no dynamics, sensor scheduling, energy accounting or delayed
command execution, and requires neither NumPy nor Torch.
