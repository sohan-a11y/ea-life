import unittest
from broker import TOPICS, retrieve


class BrokerSafetyTests(unittest.TestCase):
    def test_topic_allowlist_is_fixed_and_nonempty(self):
        self.assertIn("curiosity", TOPICS)
        self.assertTrue(all(isinstance(key, str) and isinstance(value, str) for key, value in TOPICS.items()))

    def test_arbitrary_topic_is_rejected_before_network_access(self):
        with self.assertRaises(ValueError):
            retrieve("https://example.invalid/should-not-be-requested")


if __name__ == "__main__":
    unittest.main()
