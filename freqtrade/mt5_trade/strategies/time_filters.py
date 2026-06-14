from __future__ import annotations

from datetime import UTC, datetime, time


def _parse_sessions(values: list[str]) -> list[tuple[time, time]]:
    sessions: list[tuple[time, time]] = []
    for value in values:
        start, separator, end = value.partition("-")
        if not separator:
            raise ValueError(f"Invalid session {value!r}. Expected 'HH:MM-HH:MM'.")
        sessions.append((_parse_time(start), _parse_time(end)))
    return sessions


def _parse_time(value: str) -> time:
    hour, separator, minute = value.strip().partition(":")
    if not separator:
        raise ValueError(f"Invalid time {value!r}. Expected 'HH:MM'.")
    return time(hour=int(hour), minute=int(minute))


def _time_in_range(value: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def _parse_blackout_windows(values: list[dict[str, str]]) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    for value in values:
        start = _parse_datetime(value["from"])
        end = _parse_datetime(value["to"])
        if start > end:
            raise ValueError("blackout_windows 'from' must be earlier than 'to'.")
        windows.append((start, end))
    return windows


def _parse_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
