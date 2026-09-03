import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from src.search.calibrate_client import CalibrateTraceClient, TRACE_URL


class FakeResponse:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class CalibrateTraceClientTests(unittest.TestCase):
    def test_sends_expected_trace_payload(self):
        client = CalibrateTraceClient("trace-secret", "agent-123", 1.5)
        with patch("src.search.calibrate_client.urlopen", return_value=FakeResponse()) as urlopen:
            delivered = client.send("Find a funny animal story", {"response": {"message": "Try Fox."}})

        self.assertTrue(delivered)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, TRACE_URL)
        self.assertEqual(request.get_header("X-api-key"), "trace-secret")
        self.assertEqual(json.loads(request.data), {
            "agent_id": "agent-123",
            "input": "Find a funny animal story",
            "output": {"response": {"message": "Try Fox."}},
        })

    def test_delivery_errors_are_ignored(self):
        client = CalibrateTraceClient("trace-secret", "agent-123", 1.5)
        with patch("src.search.calibrate_client.urlopen", side_effect=URLError("offline")):
            self.assertFalse(client.send("request", {"response": {}}))

    def test_missing_settings_disable_tracing(self):
        settings = SimpleNamespace(calibrate_api_key=None, calibrate_agent_id="agent-123", calibrate_trace_timeout_seconds=2)
        self.assertIsNone(CalibrateTraceClient.from_settings(settings))


if __name__ == "__main__":
    unittest.main()
