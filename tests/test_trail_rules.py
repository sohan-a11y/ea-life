"""Tests for the replaceable trail rules (NCA experiment)."""
import hashlib
import json
import unittest

from ea_life.engine import Config, Simulation

try:
    import numpy as np
    from ea_life import trail_rules as TR
except ImportError:          # numpy is needed only for the trail-rule experiments
    np = TR = None

SMALL = dict(grid_size=32, initial_population=4, max_population=6, max_total_organisms=20, scenario="forage",
             latent_model_enabled=False, imagination_enabled=False)


def fingerprint(sim):
    state = [sim.resource_trails, sim.danger_trails, sim.resources, [(a["x"], a["y"], a["energy"]) for a in sim.agents]]
    return hashlib.sha256(json.dumps(state).encode()).hexdigest()


def run(rule=None, ticks=80, seed=11):
    sim = Simulation(Config(**SMALL), seed=seed)
    if rule is not None:
        sim.trail_rule = rule
    for _ in range(ticks):
        sim.step()
    return sim


@unittest.skipIf(TR is None, "numpy not installed")
class TrailRuleTests(unittest.TestCase):
    def test_numpy_decay_equals_the_engine_rule_bit_for_bit(self):
        self.assertEqual(fingerprint(run()), fingerprint(run(TR.DecayRule())))

    def test_learned_rules_at_zero_equal_the_engine_rule_bit_for_bit(self):
        reference = fingerprint(run())
        self.assertEqual(reference, fingerprint(run(TR.NCARule([0.0] * TR.NCARule.DIM))))
        self.assertEqual(reference, fingerprint(run(TR.LinearRule([0.0] * TR.LinearRule.DIM))))

    def test_a_learned_rule_changes_the_world(self):
        theta = np.random.default_rng(3).normal(0.0, 0.3, TR.NCARule.DIM)
        self.assertNotEqual(fingerprint(run()), fingerprint(run(TR.NCARule(theta))))

    def test_trails_stay_in_range(self):
        sim = run(TR.NCARule(np.random.default_rng(0).normal(0.0, 0.5, TR.NCARule.DIM)), ticks=40)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in sim.resource_trails + sim.danger_trails))

    def test_perception_has_four_features_per_channel(self):
        X = np.random.default_rng(1).random((3, 5, 5))
        P = TR.perceive(X)
        self.assertEqual(P.shape, (12, 5, 5))
        np.testing.assert_array_equal(P[:3], X)
        np.testing.assert_allclose(TR.perceive(np.ones((1, 5, 5)))[1:, 1:-1, 1:-1], 0.0)   # flat field: no gradient inside


if __name__ == "__main__":
    unittest.main()
