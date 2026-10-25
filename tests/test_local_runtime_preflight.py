import hashlib
import json
import unittest
from pathlib import Path


PACKET = Path(__file__).resolve().parents[1] / "fixtures/local_runtime_preflight_v1.json"
PACKET_SHA256 = "6f879e8ee1bca440306af91c6a6169c97be2c1473acffc0c1882cc20bf9a7bbc"
TASK_IDS = (
    "exact-format-control",
    "factual-constant",
    "arithmetic",
    "python-comprehension",
    "synthetic-long-context",
)


class LocalRuntimePreflightPacketTests(unittest.TestCase):
    def test_v1_packet_bytes_and_contract_are_frozen(self):
        data = PACKET.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), PACKET_SHA256)
        packet = json.loads(data)
        self.assertEqual(packet["packet_id"], "local-runtime-preflight-v1")
        self.assertIn("not E-002 evidence", packet["purpose"])
        self.assertEqual(
            packet["scoring"], "exact visible-text match after no normalization"
        )
        tasks = packet["tasks"]
        self.assertEqual(tuple(task["id"] for task in tasks), TASK_IDS)
        self.assertEqual(len({task["id"] for task in tasks}), len(tasks))
        self.assertTrue(all(task["oracle"]["type"] == "exact_text" for task in tasks))

    def test_long_context_oracle_is_synthetic_and_uniquely_addressed(self):
        packet = json.loads(PACKET.read_bytes())
        task = next(task for task in packet["tasks"] if task["id"] == "synthetic-long-context")
        self.assertIn("Record 137: E53D2E3C4489", task["prompt"])
        self.assertEqual(task["prompt"].count("Record 137:"), 1)
        self.assertEqual(task["oracle"]["expected"], "E53D2E3C4489")


if __name__ == "__main__":
    unittest.main()
