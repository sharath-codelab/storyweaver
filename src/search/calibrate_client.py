"""Best-effort delivery of recommendation traces to Calibrate."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import SearchSettings


TRACE_URL = "https://api.calibrate.artpark.ai/traces"


class CalibrateTraceClient:
    def __init__(self, api_key: str, agent_id: str, timeout_seconds: float):
        self.api_key = api_key
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: SearchSettings) -> "CalibrateTraceClient | None":
        if not settings.calibrate_api_key or not settings.calibrate_agent_id:
            return None
        return cls(
            settings.calibrate_api_key,
            settings.calibrate_agent_id,
            settings.calibrate_trace_timeout_seconds,
        )

    def send(self, user_input: str, output: dict[str, Any]) -> bool:
        """Send a trace without allowing delivery failures to affect the API."""
        payload = json.dumps({
            "agent_id": self.agent_id,
            "input": user_input,
            "output": output,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            TRACE_URL,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return 200 <= response.status < 300
        except (HTTPError, URLError, OSError):
            return False
