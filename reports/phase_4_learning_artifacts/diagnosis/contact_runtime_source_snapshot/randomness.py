"""Named random streams that never depend on Python's salted hash or global RNG."""

import hashlib
import random


def seeded_rng(seed: int, stream: str) -> random.Random:
    if type(seed) is not int or not isinstance(stream, str) or not stream:
        raise ValueError("seed must be an integer and stream a nonempty string")
    digest = hashlib.sha256(f"ares.seed.v1:{seed}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))

