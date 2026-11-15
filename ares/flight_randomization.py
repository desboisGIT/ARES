"""Dependency-free, episode-seeded profile sampling and coverage auditing."""

from copy import deepcopy
import hashlib
import random

from .contracts import validate_profile


_INTEGER_PARAMETERS = frozenset({"depth_width", "depth_height", "depth_edge_width_px"})


def sample_flight_profile(profile, seed, mode="random"):
    """Copy a valid profile, replacing every nominal with an episode sample.

    Parameter-local SHA-256 seeds make results independent of dictionary order.
    Ranges, identity and schema stay unchanged. Integer ranges are inclusive;
    vector ranges are sampled independently in component order.
    """
    validate_profile(profile)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if mode not in ("random", "low", "high", "nominal"):
        raise ValueError("mode must be random, low, high or nominal")
    sampled = deepcopy(profile)
    for name, parameter in sampled["parameters"].items():
        vector = isinstance(parameter["nominal"], list)
        bounds = parameter["range"] if vector else [parameter["range"]]
        values = []
        for index, (low, high) in enumerate(bounds):
            digest = hashlib.sha256(
                f"ares.flight_randomization.v1:{seed}:{name}:{index}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest, "big"))
            if mode == "nominal":
                value = parameter["nominal"][index] if vector else parameter["nominal"]
            elif mode == "low":
                value = low
            elif mode == "high":
                value = high
            elif name in _INTEGER_PARAMETERS:
                value = rng.randint(low, high)
            else:
                value = rng.uniform(low, high)
            values.append(value)
        parameter["nominal"] = values if vector else values[0]
    validate_profile(sampled)
    return sampled


def sampled_profile_values(profile):
    """Return every sampled nominal as an independent, JSON-serializable map."""
    validate_profile(profile)
    return {name: deepcopy(parameter["nominal"])
            for name, parameter in profile["parameters"].items()}


def randomization_manifest(profiles):
    """Audit observed samples against shared declared ranges, component by component.

    Accept a profile or a nonempty iterable of profiles. Different schema versions,
    parameter sets or ranges are rejected: coverage must refer to one experiment.
    Endpoint coverage is exact and is intended for explicit low/high probes; random
    floating point samples are not expected to hit either endpoint.
    """
    profiles = [profiles] if isinstance(profiles, dict) else list(profiles)
    if not profiles:
        raise ValueError("profiles must be nonempty")
    for profile in profiles:
        validate_profile(profile)
    reference = profiles[0]
    for profile in profiles[1:]:
        if (profile["schema_version"] != reference["schema_version"]
                or profile["parameters"].keys() != reference["parameters"].keys()
                or any(profile["parameters"][name]["range"] != parameter["range"]
                       for name, parameter in reference["parameters"].items())):
            raise ValueError("profiles must share schema, parameters and ranges")
    parameters = {}
    for name in sorted(reference["parameters"]):
        parameter = reference["parameters"][name]
        vector = isinstance(parameter["nominal"], list)
        bounds = parameter["range"] if vector else [parameter["range"]]
        components = []
        for index, (low, high) in enumerate(bounds):
            values = [p["parameters"][name]["nominal"][index] if vector
                      else p["parameters"][name]["nominal"] for p in profiles]
            components.append({"index": index, "range": [low, high],
                               "observed_min": min(values), "observed_max": max(values),
                               "within_bounds": all(low <= v <= high for v in values),
                               "low_endpoint_seen": low in values,
                               "high_endpoint_seen": high in values})
        parameters[name] = {"dimensions": len(bounds),
                            "integer": name in _INTEGER_PARAMETERS,
                            "components": components}
    components = [component for p in parameters.values() for component in p["components"]]
    return {"schema_version": "ares.randomization_manifest.v1",
            "profile_schema_version": reference["schema_version"],
            "sample_count": len(profiles), "parameter_count": len(parameters),
            "dimension_count": len(components), "parameters": parameters,
            "all_within_bounds": all(c["within_bounds"] for c in components),
            "all_endpoints_seen": all(c["low_endpoint_seen"] and c["high_endpoint_seen"]
                                      for c in components)}
