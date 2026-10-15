import json
import random
import tempfile
import unittest
from pathlib import Path

from ea_life.engine import ACTIONS, Config, GENE_BOUNDS, Simulation, curiosity_score, mutate_genome, new_genome
from ea_life.latent_world_model import FEATURE_DIM, NeuralLatentWorldModel


class EngineTests(unittest.TestCase):
    def test_configuration_bounds(self):
        with self.assertRaises(ValueError):
            Config(grid_size=8)
        with self.assertRaises(ValueError):
            Config(scenario="arbitrary_task")
        self.assertEqual(Config().scenario, "mixed")
        with self.assertRaises(ValueError):
            Config(imagination_horizon=13)
        with self.assertRaises(ValueError):
            Config(imagination_beam_width=9)

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
        self.assertEqual(snap["schema_version"], 3)
        self.assertIn("resource_trails", snap)
        self.assertIn("danger_trails", snap)
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

    def test_evolution_can_be_disabled(self):
        cfg = Config(grid_size=48, initial_population=4, max_population=8,
                     evolution_enabled=False)
        sim = Simulation(cfg, seed=22)
        for agent in sim.agents:
            agent["energy"] = 100.0
            agent["genome"]["reproduction_threshold"] = 50.0
            agent["last_birth_tick"] = -1000
        sim.step()
        self.assertEqual(len(sim.agents), 4)
        self.assertEqual(sim.total_births, 0)

    def test_checkpoint_rejects_incompatible_config(self):
        first = Simulation(Config(grid_size=48, initial_population=4, max_population=8), seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            first.save_checkpoint(path)
            data = json.loads(path.read_text())
            incompatible = Simulation(Config(grid_size=48, initial_population=4, max_population=9), seed=5)
            with self.assertRaises(ValueError):
                incompatible.load_state_dict(data)

    def test_predictive_world_model_learns_and_stays_bounded(self):
        cfg = Config(grid_size=48, initial_population=4, max_population=8,
                     max_world_model_states=16)
        sim = Simulation(cfg, seed=101)
        for _ in range(36):
            sim.step()
        self.assertGreater(sim.metrics()["world_model_entries"], 0)
        self.assertGreater(sim.metrics()["world_model_predictions"], 0)
        for agent in sim.agents:
            self.assertLessEqual(len(agent.get("world_model", {})), 16)
            for row in agent.get("world_model", {}).values():
                self.assertLessEqual(len(row["next_counts"]), 24)
        self.assertGreaterEqual(sim.metrics()["world_model_next_state_accuracy"], 0.0)
        self.assertLessEqual(sim.metrics()["world_model_next_state_accuracy"], 1.0)

    def test_world_model_and_collective_mechanisms_can_be_ablated(self):
        cfg = Config(grid_size=48, initial_population=4, max_population=8,
                     world_model_enabled=False, collective_enabled=False)
        sim = Simulation(cfg, seed=102)
        for _ in range(12):
            sim.step()
        self.assertEqual(sim.metrics()["world_model_entries"], 0)
        self.assertTrue(all(not a.get("world_model") for a in sim.agents))
        self.assertEqual(sum(sim.resource_trails), 0.0)
        self.assertEqual(sum(sim.danger_trails), 0.0)

    def test_collective_resource_trails_are_shared_and_visible(self):
        cfg = Config(grid_size=48, initial_population=4, max_population=8,
                     evolution_enabled=False)
        sim = Simulation(cfg, seed=103)
        for agent in sim.agents:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x, y = agent["x"] + dx, agent["y"] + dy
                    if 0 <= x < cfg.grid_size and 0 <= y < cfg.grid_size:
                        sim.resources[sim._idx(x, y)] = .95
        for _ in range(4):
            sim.step()
        self.assertGreater(sum(sim.resource_trails), 0.0)
        self.assertGreater(sim.metrics()["collective_signal_coverage"], 0.0)
        snapshot = sim.public_snapshot()
        self.assertEqual(len(snapshot["resource_trails"]), cfg.grid_size)

    def test_neural_latent_model_learns_and_imagines(self):
        model = NeuralLatentWorldModel(seed=123, replay_capacity=128)
        for i in range(64):
            obs = [((i * (j + 3)) % 101) / 100.0 for j in range(FEATURE_DIM)]
            nxt = [max(0.0, min(1.0, value + (0.02 if j % 2 else -0.015)))
                   for j, value in enumerate(obs)]
            model.observe_transition(obs, i % len(ACTIONS), nxt, (i % 7) / 3.0 - 1.0)
        self.assertTrue(model.ready)
        self.assertEqual(len(model.encode([0.5] * FEATURE_DIM)), 8)
        decoded = model.reconstruct([0.5] * FEATURE_DIM)
        self.assertEqual(len(decoded), FEATURE_DIM)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in decoded))
        plan = model.imagine([0.5] * FEATURE_DIM, horizon=4, beam_width=3,
                             legal_first_actions=[0, 4, 8])
        self.assertIsNotNone(plan)
        self.assertIn(plan["action"], [0, 4, 8])
        self.assertEqual(len(plan["path"]), 4)
        self.assertGreater(plan["predictions"], 0)
        self.assertLessEqual(len(model.replay), 128)
        restored = NeuralLatentWorldModel(seed=999, replay_capacity=128)
        import copy
        state = json.loads(json.dumps(model.state_dict()))
        restored.load_state_dict(state)
        self.assertEqual(restored.transitions_seen, model.transitions_seen)
        self.assertEqual(restored.encode([0.2] * FEATURE_DIM), model.encode([0.2] * FEATURE_DIM))

    def test_simulation_trains_latent_model_and_uses_imagination(self):
        cfg = Config(grid_size=32, initial_population=4, max_population=4,
                     evolution_enabled=False, imagination_horizon=3,
                     imagination_beam_width=2)
        sim = Simulation(cfg, seed=555)
        for _ in range(20):
            sim.step()
        metrics = sim.metrics()
        self.assertTrue(metrics["latent_model_ready"])
        self.assertGreater(metrics["latent_transitions_seen"], 48)
        self.assertGreater(metrics["latent_imagination_calls"], 0)
        self.assertGreater(metrics["latent_imagination_predictions"], 0)
        self.assertEqual(metrics["latent_last_plan_horizon"], 3)
        snapshot = sim.public_snapshot()
        self.assertEqual(snapshot["schema_version"], 4)
        self.assertNotIn("latent_world_model", snapshot)

    def test_latent_model_and_imagination_can_be_disabled_separately(self):
        cfg = Config(grid_size=32, initial_population=4, max_population=4,
                     evolution_enabled=False, latent_model_enabled=False,
                     imagination_enabled=False)
        sim = Simulation(cfg, seed=556)
        for _ in range(12):
            sim.step()
        metrics = sim.metrics()
        self.assertEqual(metrics["latent_transitions_seen"], 0)
        self.assertEqual(metrics["latent_imagination_calls"], 0)

    def test_action_space_has_stay_and_eight_moves(self):
        self.assertEqual(len(ACTIONS), 9)
        self.assertIn((0, 0), ACTIONS)


if __name__ == "__main__":
    unittest.main()
