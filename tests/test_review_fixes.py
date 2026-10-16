"""Tests for the fixes made in the 16 Oct 2026 review."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ea_life.engine import Config, Simulation

ROOT = Path(__file__).resolve().parents[1]
SMALL = dict(grid_size=32, initial_population=4, max_population=4, evolution_enabled=False)


class ReviewFixTests(unittest.TestCase):
    def test_dead_organisms_drop_their_world_model(self):
        sim = Simulation(Config(**SMALL), seed=304)
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
        sim = Simulation(Config(**SMALL), seed=305)
        for _ in range(10):
            sim.step()
        expected = sum(a["reward_total"] for a in sim.agents) / sum(a["age"] for a in sim.agents)
        self.assertAlmostEqual(sim.metrics()["reward_per_agent_tick"], round(expected, 4))

    def test_analysis_runs_on_the_ablation_suite(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ablation.csv"
            subprocess.run([sys.executable, str(ROOT / "experiments.py"), "--ablation-suite", "--seeds", "2",
                            "--ticks", "15", "--grid-size", "32", "--population", "4", "--max-population", "4",
                            "--max-organisms", "4", "--output", str(out)], check=True, capture_output=True,
                           env=env, cwd=str(ROOT))
            with out.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)
            report = subprocess.run([sys.executable, str(ROOT / "analyze.py"), str(out)], check=True,
                                    capture_output=True, text=True, env=env, cwd=str(ROOT)).stdout
            self.assertIn("Reference condition: baseline", report)
            self.assertIn("reward_per_agent_tick", report)


if __name__ == "__main__":
    unittest.main()
