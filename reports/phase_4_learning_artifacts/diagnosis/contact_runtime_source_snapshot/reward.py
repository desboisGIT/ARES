"""Dense TRAINING signals, deliberately separate from the pass/fail scorer."""

import math


def dense_reward(*, new_correct_voxels: int, new_rooms: int, elapsed_s: float,
                 collided: bool = False) -> float:
    """No reward value constitutes a passing episode or replaces the scorer."""
    if type(new_correct_voxels) is not int or new_correct_voxels < 0:
        raise ValueError("new_correct_voxels must be a nonnegative integer")
    if type(new_rooms) is not int or new_rooms < 0:
        raise ValueError("new_rooms must be a nonnegative integer")
    if (isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float))
            or not math.isfinite(elapsed_s) or elapsed_s < 0):
        raise ValueError("elapsed_s must be finite and nonnegative")
    if type(collided) is not bool:
        raise ValueError("collided must be boolean")
    return 0.01 * new_correct_voxels + new_rooms - 0.01 * elapsed_s - 10 * collided
