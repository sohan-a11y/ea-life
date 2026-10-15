import json
import random
import tempfile
import unittest
from pathlib import Path

from ea_life.engine import ACTIONS, Config, GENE_BOUNDS, Simulation, curiosity_score, mutate_genome, new_genome


class EngineTests(unittest.TestCase):
    def test_configuration_bounds(self):
        with self.assertRaises(ValueError):
            Config(grid_size=8)
        with self.assertRaises(ValueError):
            Config(scenario="arbitrary_task")
        self.assertEqual(Config().scenario, "mixed")

    def test_world_bounds_hazards_and_beacons(self):
        sim = Simulation(Config(grid_size=48, initial_population=4, max_population=8), seed=4)
        self.assertEqual(len(sim.resources), 48 * 48)
        self.assertTrue(all(0 <= v <= 1 for v in sim.resources))
        self.assertTrue(all(0 <= v <= 1 for v in sim.hazards))
        self.assertTrue(any(v > 0 for v in sim.hazards))
        self.assertEqual(len(sim.beacons), 4)

    def test_mutation_is_bounded_and_changes_genome(self):
        rng = random.Random(77)
        parent = new_genome(rng)
        child = mutate_genome(parent, rng)
        self.assertNotEqual(parent, child)
        for gene, (low, high) in GENE_BOUNDS.items():
            self.assertGreaterEqual(child[gene], low)
            self.assertLessEqual(child[gene], high)

    def test_curiosity_is_bounded(self):
        sim = Simulation(Config(grid_size=48, initial_population=4, max_population=8), seed=2)
        score = curiosity_score(sim.agents[0]["genome"], 1.0, 0.8, 0.2, 0.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_step_updates_learning_and_bounded_memory(self):
        cfg = Config(grid_size=48, initial_population=4, max_population=8, max_memory=8)
        sim = Simulation(cfg, seed=42)
        for _ in range(24):
            sim.step()
        self.assertEqual(sim.tick, 24)
        self.assertTrue(all(len(a["memory"]) <= 8 for a in sim.agents))
        self.assertTrue(all(len(a["q_table"]) <= cfg.max_q_states for a in sim.agents))
        self.assertGreater(sum(a["prediction_count"] for a in sim.agents), 0)

    def test_public_snapshot_hides_private_internal_state(self):
        sim = Simulation(Config(grid_size=48, initial_population=4, max_population=8), seed=8)
        snap = sim.public_snapshot()
        self.assertEqual(snap["schema_version"], 2)
        self.assertEqual(len(snap["resources"]), 48)
        self.assertNotIn("genome", snap["agents"][0])
        self.assertNotIn("q_table", snap["agents"][0])
        self.assertNotIn("memory", snap["agents"][0])

    def test_checkpoint_round_trip_keeps_rng_sequence(self):
        cfg = Config(grid_size=48, initial_population=4, max_population=8)
        first = Simulation(cfg, seed=12)
        for _ in range(5):
            first.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            first.save_checkpoint(path)
            data = json.loads(path.read_text())
            second = Simulation(cfg, seed=999)
            second.load_state_dict(data)
            self.assertEqual(second.tick, first.tick)
            second.step()
            first.step()
            self.assertEqual(second.public_snapshot()["tick"], first.public_snapshot()["tick"])
            self.assertEqual(second.agents[0]["x"], first.agents[0]["x"])
            self.assertEqual(second.agents[0]["y"], first.agents[0]["y"])

    def test_roles_and_task_scenario_map(self):
        sim = Simulation(Config(grid_size=48, initial_population=4, max_population=8), seed=9)
        self.assertEqual([a["role"] for a in sim.agents], ["forager", "explorer", "navigator", "avoider"])
        self.assertEqual(sim._task(sim.agents[2]), "navigator")
        forage = Simulation(Config(grid_size=48, initial_population=4, max_population=8, scenario="avoid_hazards"), seed=9)
        self.assertTrue(all(forage._task(a) == "avoider" for a in forage.agents))

    def test_action_space_has_stay_and_eight_moves(self):
        self.assertEqual(len(ACTIONS), 9)
        self.assertIn((0, 0), ACTIONS)


if __name__ == "__main__":
    unittest.main()
