"""Load probe_all_results.json for endpoint health metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nba_fit.config.settings import Settings, get_settings


@dataclass(frozen=True)
class EndpointHealth:
    endpoint: str
    success: bool
    has_data: bool
    total_rows: int
    api_family: str
    failure_reason: str | None
    elapsed_sec: float | None

    @property
    def status(self) -> str:
        if not self.success:
            return "FAIL"
        if self.has_data:
            return "OK"
        return "EMPTY"


@dataclass
class ProbeRegistry:
    """In-memory view of local endpoint probe results."""

    meta: dict[str, Any]
    endpoints: dict[str, EndpointHealth]

    @classmethod
    def load(cls, path: Path | None = None, settings: Settings | None = None) -> ProbeRegistry:
        settings = settings or get_settings()
        path = path or settings.probe_results_path
        if not path.exists():
            raise FileNotFoundError(
                f"Probe results not found at {path}. Run probe_all_nba_endpoints.py first."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        meta = raw.get("meta", {})
        endpoints: dict[str, EndpointHealth] = {}
        for row in raw.get("results", []):
            name = row["endpoint"]
            endpoints[name] = EndpointHealth(
                endpoint=name,
                success=bool(row.get("success")),
                has_data=bool(row.get("has_data", row.get("total_rows", 0) > 0)),
                total_rows=int(row.get("total_rows") or 0),
                api_family=str(row.get("api_family", "stats")),
                failure_reason=row.get("failure_reason"),
                elapsed_sec=row.get("elapsed_sec"),
            )
        return cls(meta=meta, endpoints=endpoints)

    def summary(self) -> dict[str, int]:
        counts = {"OK": 0, "EMPTY": 0, "FAIL": 0}
        for ep in self.endpoints.values():
            counts[ep.status] += 1
        return counts

    def row_counts(self) -> dict[str, int]:
        return {name: ep.total_rows for name, ep in self.endpoints.items()}

    def get(self, endpoint: str) -> EndpointHealth | None:
        return self.endpoints.get(endpoint)

    def essential_status(self, essential_names: tuple[str, ...]) -> list[EndpointHealth]:
        return [
            self.endpoints[name]
            for name in essential_names
            if name in self.endpoints
        ]
