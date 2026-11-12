import random
import unittest

from ares.randomness import seeded_rng
from ares.reward import dense_reward


class RandomnessTests(unittest.TestCase):
    def sample(self, seed, stream):
        rng = seeded_rng(seed, stream)
        return [rng.random() for _ in range(10)]

    def test_reproducible_and_independent_streams(self):
        self.assertEqual(self.sample(42, "scene"), self.sample(42, "scene"))
        self.assertNotEqual(self.sample(42, "scene"), self.sample(43, "scene"))
        self.assertNotEqual(self.sample(42, "scene"), self.sample(42, "sensor"))

    def test_global_rng_untouched(self):
        state = random.getstate()
        self.sample(1, "scene")
        self.assertEqual(state, random.getstate())

    def test_rejects_invalid_inputs(self):
        for seed, stream in [(True, "x"), (1.0, "x"), (None, "x"),
                             (float("nan"), "x"), (1, ""), (1, 2)]:
            with self.subTest(seed=seed, stream=stream), self.assertRaises(ValueError):
                seeded_rng(seed, stream)


class DenseRewardTests(unittest.TestCase):
    def reward(self, **changes):
        args = dict(new_correct_voxels=0, new_rooms=0, elapsed_s=0)
        args.update(changes)
        return dense_reward(**args)

    def test_training_signal_and_collision_penalty(self):
        self.assertGreater(self.reward(new_correct_voxels=1), self.reward())
        self.assertGreater(self.reward(new_rooms=1), self.reward())
        self.assertLess(self.reward(elapsed_s=1), self.reward())
        self.assertLess(self.reward(collided=True), self.reward())

    def test_positive_reward_does_not_certify_pass(self):
        # Even a collided episode can earn positive shaping reward. The scorer
        # must independently enforce its zero-collision passing condition.
        reward = self.reward(new_correct_voxels=2000, collided=True)
        self.assertIs(type(reward), float)
        self.assertGreater(reward, 0)

    def test_invalid_inputs(self):
        invalid = {
            "new_correct_voxels": [-1, True, 1.0, None, float("inf")],
            "new_rooms": [-1, False, 1.0, None, float("nan")],
            "elapsed_s": [-1, True, "1", None, float("inf"), float("-inf"), float("nan")],
            "collided": [0, 1, "false", None],
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.reward(**{name: value})


if __name__ == "__main__":
    unittest.main()
