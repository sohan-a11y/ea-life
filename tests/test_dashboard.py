import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dashboard


class DashboardStateTests(unittest.TestCase):
    def test_build_state_passes_through_collective_trail_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = {
                "schema_version": 4, "tick": 7, "grid_size": 2,
                "resources": [[0.1, 0.2], [0.3, 0.4]],
                "hazards": [[0.0, 0.1], [0.0, 0.0]],
                "resource_trails": [[0.5, 0.0], [0.0, 0.0]],
                "danger_trails": [[0.0, 0.7], [0.0, 0.0]],
                "beacons": [], "agents": [], "metrics": {}
            }
            (root / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            with patch.object(dashboard, "STATE_DIR", root), patch.object(dashboard, "CONTROL_DIR", root / "control"):
                state = dashboard.build_state()
            self.assertEqual(state["resource_trails"], snapshot["resource_trails"])
            self.assertEqual(state["danger_trails"], snapshot["danger_trails"])

    def test_event_response_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshot.json").write_text(json.dumps({
                "tick": 10, "grid_size": 32, "agents": [], "metrics": {}, "resources": [], "hazards": []
            }), encoding="utf-8")
            (root / "events.jsonl").write_text(
                "".join(json.dumps({"seq": i, "tick": i, "type": "test"}) + "\n" for i in range(1, 1501)),
                encoding="utf-8")
            with patch.object(dashboard, "STATE_DIR", root), patch.object(dashboard, "CONTROL_DIR", root / "control"):
                state = dashboard.build_state()
            self.assertLessEqual(len(state["events"]), 100)
            self.assertEqual(state["events"][-1]["seq"], 1500)


if __name__ == "__main__":
    unittest.main()
