"""In-memory fake repositories for testing without a database."""

from typing import Any

from app.utils.helpers import generate_uuid, get_current_time_iso


class FakeMatchResultRepository:
    """In-memory match result repository for unit tests."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        match_id = data.get("id", generate_uuid())
        data["id"] = match_id
        data.setdefault("created_at", get_current_time_iso())
        data.setdefault("updated_at", get_current_time_iso())
        self._store[match_id] = data
        return data

    async def get_by_id(self, match_id: str) -> dict[str, Any] | None:
        return self._store.get(match_id)

    async def get_by_user(
        self,
        user_id: str,
        entity_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        results = [m for m in self._store.values() if m.get("user_id") == user_id]
        if entity_type:
            results = [m for m in results if m.get("entity_type") == entity_type]
        return results[offset : offset + limit]

    async def count_by_user(self, user_id: str, entity_type: str | None = None) -> int:
        results = [m for m in self._store.values() if m.get("user_id") == user_id]
        if entity_type:
            results = [m for m in results if m.get("entity_type") == entity_type]
        return len(results)

    async def delete(self, match_id: str) -> int:
        if match_id in self._store:
            del self._store[match_id]
            return 1
        return 0


class FakeAttributionReportRepository:
    """In-memory attribution report repository for unit tests."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        report_id = data.get("id", generate_uuid())
        data["id"] = report_id
        data.setdefault("created_at", get_current_time_iso())
        self._store[data["match_id"]] = data
        return data

    async def get_by_match_id(self, match_id: str) -> dict[str, Any] | None:
        return self._store.get(match_id)

    async def get_by_id(self, report_id: str) -> dict[str, Any] | None:
        for report in self._store.values():
            if report.get("id") == report_id:
                return report
        return None


class FakeScoringHistoryRepository:
    """In-memory scoring history repository for unit tests."""

    def __init__(self):
        self._store: list[dict[str, Any]] = []

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = data.get("id", generate_uuid())
        data.setdefault("created_at", get_current_time_iso())
        self._store.append(data)
        return data

    async def get_by_match_id(self, match_id: str) -> list[dict[str, Any]]:
        return [h for h in self._store if h.get("match_id") == match_id]
