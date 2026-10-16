import json
import tempfile
import unittest
from pathlib import Path

from runner import append_event_batch


class EventLogTests(unittest.TestCase):
    def test_event_log_rotates_and_remains_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text("".join(json.dumps({"seq": i}) + "\n" for i in range(20)), encoding="utf-8")
            append_event_batch(path, [{"seq": 100}, {"seq": 101}], max_bytes=128)
            rotated = Path(tmp) / "events.previous.jsonl"
            self.assertTrue(rotated.exists())
            self.assertLessEqual(path.stat().st_size, 128)
            parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["seq"] for row in parsed], [100, 101])

    def test_oversized_batch_keeps_only_newest_events_that_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            events = [{"seq": i, "message": "x" * 30} for i in range(20)]
            append_event_batch(path, events, max_bytes=256)
            self.assertLessEqual(path.stat().st_size, 256)
            parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(parsed)
            self.assertEqual(parsed[-1]["seq"], 19)


if __name__ == "__main__":
    unittest.main()
