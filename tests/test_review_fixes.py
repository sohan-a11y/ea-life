"""Tests for the fixes made in the 16 Oct 2026 review."""
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ea_life.engine import ROLES, Config, Simulation
from ea_life.latent_world_model import FEATURE_DIM, NeuralLatentWorldModel

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(grid_size=32, initial_population=4, max_population=4, evolution_enabled=False)


class ReviewFixTests(unittest.TestCase):
    def test_latent_features_include_the_task(self):
        sim = Simulation(Config(**SMALL), seed=301)
        for agent in sim.agents:
            features = sim._latent_features(agent)
            self.assertEqual(len(features), FEATURE_DIM)
            self.assertEqual(FEATURE_DIM, 18)
            one_hot = features[-len(ROLES):]
            self.assertEqual(sum(one_hot), 1.0)
            self.assertEqual(ROLES[one_hot.index(1.0)], sim._task(agent))
            self.assertTrue(all(0.0 <= value <= 1.0 for value in features))

    def test_replayed_samples_do_not_enter_reported_errors(self):
        model = NeuralLatentWorldModel(seed=5)
        sample = {"obs": [0.2] * FEATURE_DIM, "action": 3, "next_obs": [0.4] * FEATURE_DIM, "reward": 1.5}
        model.observe_transition(sample["obs"], 3, sample["next_obs"], 1.5)
        before = (model.reward_mae_ema, model.transition_loss_ema, model.reconstruction_loss_ema)
        model._train_sample(sample, record=False)
        self.assertEqual(before, (model.reward_mae_ema, model.transition_loss_ema, model.reconstruction_loss_ema))

    def test_evaluate_scores_without_training(self):
        model = NeuralLatentWorldModel(seed=6)
        samples = []
        for i in range(80):
            obs = [((i * (j + 1)) % 50) / 50.0 for j in range(FEATURE_DIM)]
            nxt = [min(1.0, v + 0.01) for v in obs]
            model.observe_transition(obs, i % 9, nxt, (i % 5) - 2.0)
            samples.append({"obs": obs, "action": i % 9, "next_obs": nxt, "reward": (i % 5) - 2.0})
        weights = copy.deepcopy(model.state_dict())
        result = model.evaluate(samples[-20:])
        self.assertEqual(result["heldout_transitions"], 20)
        for key in ("heldout_reward_mae", "heldout_constant_reward_mae", "heldout_latent_mse",
                    "heldout_persistence_latent_mse"):
            self.assertIn(key, result)
        self.assertEqual(weights["encoder_w"], model.state_dict()["encoder_w"])
        self.assertEqual(weights["training_updates"], model.training_updates)

    def test_old_or_misshapen_latent_checkpoints_are_rejected(self):
        model = NeuralLatentWorldModel(seed=7)
        state = json.loads(json.dumps(model.state_dict()))
        old = dict(state, version=2)
        with self.assertRaises(ValueError):
            NeuralLatentWorldModel(seed=8).load_state_dict(old)
        bad = dict(state, encoder_w=[row[:-1] for row in state["encoder_w"]])
        with self.assertRaises(ValueError):
            NeuralLatentWorldModel(seed=8).load_state_dict(bad)
        NeuralLatentWorldModel(seed=8).load_state_dict(state)

    def test_frozen_latent_model_records_heldout_transitions_without_training(self):
        sim = Simulation(Config(**SMALL, imagination_enabled=False), seed=302)
        for _ in range(15):
            sim.step()
        sim.freeze_latent_model()
        updates = sim.latent_model.training_updates
        for _ in range(5):
            sim.step()
        self.assertEqual(updates, sim.latent_model.training_updates)
        self.assertGreater(len(sim.latent_heldout), 0)
        self.assertIn("heldout_reward_mae", sim.latent_heldout_metrics())

    def test_plan_influence_is_measured_and_zero_bonus_never_promotes(self):
        for bonus, expect_promotions in ((0.0, False), (0.45, None)):
            sim = Simulation(Config(**SMALL, imagination_horizon=2, imagination_beam_width=2,
                                    imagination_bonus=bonus), seed=303)
            for _ in range(25):
                sim.step()
            metrics = sim.metrics()
            self.assertGreater(sim.plan_stats["planned"], 0)
            self.assertTrue(0.0 <= metrics["latent_plan_adoption_rate"] <= 1.0)
            self.assertTrue(0.0 <= metrics["latent_plan_promotion_rate"] <= 1.0)
            if expect_promotions is False:
                self.assertEqual(sim.plan_stats["promoted"], 0)

    def test_untrained_residual_dynamics_start_near_no_change(self):
        model = NeuralLatentWorldModel(seed=9)
        z = model.encode([0.3] * FEATURE_DIM)
        nxt, _ = model.predict(z, 4)
        self.assertLess(max(abs(a - b) for a, b in zip(z, nxt)), 0.2)

    def test_imagination_bonus_is_validated(self):
        with self.assertRaises(ValueError):
            Config(imagination_bonus=-0.1)
        with self.assertRaises(ValueError):
            Config(imagination_bonus=11.0)

    def test_dead_organisms_drop_their_world_model(self):
        sim = Simulation(Config(**SMALL, latent_model_enabled=False, imagination_enabled=False), seed=304)
        for _ in range(5):
            sim.step()
        victim = sim.agents[0]
        self.assertTrue(victim["world_model"])
        victim["age"] = sim.config.max_age - 1       # reaches the age limit during this step
        sim.step()
        self.assertFalse(victim["alive"])
        self.assertEqual(victim["world_model"], {})
        living = sum(len(a["world_model"]) for a in sim.agents if a["alive"])
        self.assertEqual(sim.metrics()["world_model_entries"], living)

    def test_reward_per_agent_tick_is_reported(self):
        sim = Simulation(Config(**SMALL, latent_model_enabled=False, imagination_enabled=False), seed=305)
        for _ in range(10):
            sim.step()
        expected = sum(a["reward_total"] for a in sim.agents) / sum(a["age"] for a in sim.agents)
        self.assertAlmostEqual(sim.metrics()["reward_per_agent_tick"], round(expected, 4))

    def test_experiment_csv_keeps_heldout_columns_and_analysis_flags_identical_arms(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "latent.csv"
            subprocess.run([sys.executable, str(ROOT / "experiments.py"), "--latent-ablation-suite", "--seeds", "2",
                            "--ticks", "20", "--heldout-ticks", "5", "--grid-size", "32", "--population", "4",
                            "--max-population", "4", "--max-organisms", "4", "--imagination-horizon", "2",
                            "--output", str(out)], check=True, capture_output=True, env=env, cwd=str(ROOT))
            with out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertIn("heldout_reward_mae", rows[0])
            self.assertEqual(rows[0]["heldout_reward_mae"], "")            # v3_control has no latent model
            self.assertNotEqual(rows[-1]["heldout_reward_mae"], "")
            report = subprocess.run([sys.executable, str(ROOT / "analyze.py"), str(out)], check=True,
                                    capture_output=True, text=True, env=env, cwd=str(ROOT)).stdout
            self.assertIn("latent_model_only (identical to reference on every seed)", report)

class DashboardComposeTests(unittest.TestCase):
    """Static checks; the CI container step tests the running stack."""

    def _dashboard_and_networks(self):
        lines = (ROOT / "compose.yaml").read_text(encoding="utf-8").splitlines()
        text = "\n".join(line for line in lines if not line.lstrip().startswith("#")) + "\n"   # ignore YAML comments
        dashboard = text[text.index("\n  dashboard:"):text.index("\n  broker:")]
        return dashboard, text[text.rindex("\nnetworks:"):]

    def test_dashboard_port_is_published_on_loopback_only(self):
        dashboard, _ = self._dashboard_and_networks()
        self.assertIn('ports: ["127.0.0.1:8000:8000"]', dashboard)

    def test_dashboard_network_can_publish_ports_without_internet_route(self):
        _, networks = self._dashboard_and_networks()
        self.assertNotIn("internal: true", networks)      # internal-only networks get no published ports
        self.assertIn('com.docker.network.bridge.enable_ip_masquerade: "false"', networks)


if __name__ == "__main__":
    unittest.main()
