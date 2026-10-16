import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from broker import TOPICS, retrieve, parse_request


class BrokerSafetyTests(unittest.TestCase):
    def test_topic_allowlist_is_fixed_and_nonempty(self):
        self.assertIn("curiosity", TOPICS)
        self.assertTrue(all(isinstance(key, str) and isinstance(value, str) for key, value in TOPICS.items()))

    def test_arbitrary_topic_is_rejected_before_network_access(self):
        with self.assertRaises(ValueError):
            retrieve("https://example.invalid/should-not-be-requested")

    def test_redirect_to_unapproved_host_is_blocked(self):
        handler = __import__("broker").AllowlistedRedirectHandler()
        request = __import__("urllib.request", fromlist=["Request"]).Request("https://en.wikipedia.org/w/api.php")
        with self.assertRaises(urllib.error.HTTPError):
            handler.redirect_request(request, None, 302, "Found", {}, "https://example.com/")

    def test_broker_rejects_oversized_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ("req_" + "a" * 32 + ".json")
            path.write_text("x" * 5000, encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_request(path)

    def test_broker_rejects_non_object_json_and_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ("req_" + "a" * 32 + ".json")
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_request(path)
            path.write_text(json.dumps({"request_id": "b" * 32, "topic_id": "curiosity"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_request(path)

    def test_broker_status_is_atomically_published(self):
        with tempfile.TemporaryDirectory() as tmp, patch("broker.RESULT_DIR", Path(tmp)):
            import broker
            broker.write_status("exhausted", 200)
            status = json.loads((Path(tmp) / "broker_status.json").read_text())
            self.assertEqual(status["state"], "exhausted")
            self.assertEqual(status["completed_requests"], 200)


if __name__ == "__main__":
    unittest.main()
