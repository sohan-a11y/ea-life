import random
import unittest
from life import ACTIONS, GRID, curiosity_score, mutate_genome, new_agent, new_genome, seed_world

class LifeCoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(7)
    def test_world_dimensions(self):
        world = seed_world(self.rng)
        self.assertEqual(len(world), GRID)
        self.assertTrue(all(len(row) == GRID for row in world))
        self.assertTrue(all(0 <= v <= 1 for row in world for v in row))
    def test_genome_has_motivational_traits(self):
        g = new_genome(self.rng)
        for key in ("resource_weight", "novelty_weight", "risk_weight", "reproduction_threshold"):
            self.assertIn(key, g)
    def test_mutation_changes_genome_and_stays_bounded(self):
        parent = new_genome(self.rng); child = mutate_genome(parent, self.rng)
        self.assertNotEqual(parent, child)
        self.assertTrue(0.1 <= child["resource_weight"] <= 4.0)
        self.assertTrue(0.03 <= child["mutation_scale"] <= 0.25)
    def test_curiosity_is_bounded(self):
        a = new_agent("a", 10, 10, self.rng)
        value = curiosity_score(a, 1.0, 0.8, 0.0, 0.1)
        self.assertTrue(0.0 <= value <= 1.0)
    def test_action_space_contains_nine_moves(self):
        self.assertEqual(len(ACTIONS), 9)
        self.assertIn((0,0), ACTIONS)

if __name__ == "__main__":
    unittest.main()
