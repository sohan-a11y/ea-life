import json
import math
import random
import tempfile
import time
import unittest
from unittest.mock import patch
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


    def test_terminal_transition_does_not_bootstrap_next_q_value(self):
        cfg = Config(grid_size=48, initial_population=1, max_population=1,
                     evolution_enabled=False, max_age=1)
        sim = Simulation(cfg, seed=111)
        agent = sim.agents[0]
        agent["energy"] = 100.0
        # Force a stay action and distinct abstract state keys to isolate the update target.
        original_q_values = sim._q_values
        q_keys = []
        def track_q_values(target_agent, key):
            q_keys.append(key)
            return original_q_values(target_agent, key)
        with patch.object(sim, "_state_key", side_effect=["old-state", "next-state"]), \
             patch.object(sim, "_choose_action", return_value=(4, 0.1, "old-state", 0.1)), \
             patch.object(sim, "_q_values", side_effect=track_q_values):
            sim.step()
        self.assertFalse(agent["alive"])
        self.assertIn("old-state", q_keys)
        self.assertNotIn("next-state", q_keys)

    def test_world_model_surprise_uses_retained_outcome_counts(self):
        sim = Simulation(Config(grid_size=48, initial_population=1, max_population=1,
                                evolution_enabled=False), seed=112)
        agent = sim.agents[0]
        for i in range(35):
            sim._update_world_model(agent, "state", 0, "next-" + str(i), float(i % 3))
        row = agent["world_model"]["state|0"]
        self.assertLessEqual(len(row["next_counts"]), 24)
        self.assertGreater(sum(row["next_counts"].values()), 0)
        sim._update_world_model(agent, "state", 0, "next-34", 1.0)
        self.assertTrue(math.isfinite(agent["world_model_surprise_sum"]))


    def test_research_requests_stop_when_broker_is_exhausted_or_stale_and_queue_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = root / "requests"
            results = root / "results"
            requests.mkdir()
            results.mkdir()
            (results / "broker_status.json").write_text(
                json.dumps({"state": "exhausted", "updated_at": time.time()}), encoding="utf-8")
            sim = Simulation(Config(grid_size=48, initial_population=1, max_population=1),
                             seed=121, request_dir=requests, result_dir=results)
            agent = sim.agents[0]
            agent["curiosity"] = 1.0
            sim._request_topic(agent)
            self.assertEqual(list(requests.glob("req_*.json")), [])

            (results / "broker_status.json").write_text(
                json.dumps({"state": "active", "updated_at": time.time() - 60}), encoding="utf-8")
            sim._request_topic(agent)
            self.assertEqual(list(requests.glob("req_*.json")), [])

            (results / "broker_status.json").write_text(
                json.dumps({"state": "active", "updated_at": time.time()}), encoding="utf-8")
            for i in range(128):
                (requests / ("req_%032x.json" % i)).write_text("{}", encoding="utf-8")
            sim._request_topic(agent)
            self.assertEqual(len(list(requests.glob("req_*.json"))), 128)

    def test_action_space_has_stay_and_eight_moves(self):
        self.assertEqual(len(ACTIONS), 9)
        self.assertIn((0, 0), ACTIONS)


if __name__ == "__main__":
    unittest.main()
