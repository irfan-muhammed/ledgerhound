"""
sentinel.monitor.schedule
==========================

Calendar-aware scheduling for the Drift Monitor: run every N seconds
(the original behaviour, unchanged and still the default) OR on specific
days of the week at a specific time of day, in a specific timezone.

Deliberately pure functions -- computing "when is the next run" is fully
unit-testable without sleeping or mocking ``time.time()``. The daemon
loop (``DriftMonitor.run_daemon``) is the only place that actually sleeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dt_time, timezone
from typing import Any, Optional

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WEEKDAY_INDEX = {d: i for i, d in enumerate(_WEEKDAYS)}


@dataclass
class ScheduleConfig:
    """How often the monitor runs.

    mode="interval" (default, backward-compatible with the original
    ``schedule_interval_seconds`` field): run every ``interval_seconds``.

    mode="calendar": run on specific ``days`` (weekday abbreviations,
    e.g. ["mon", "thu"]; empty list = every day) at ``time_of_day``
    ("HH:MM", 24h) in ``timezone`` (IANA name, e.g. "Asia/Kolkata").
    """

    mode: str = "interval"
    interval_seconds: int = 3600
    days: list[str] = field(default_factory=list)
    time_of_day: str = "06:00"
    timezone: str = "UTC"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ScheduleConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.mode not in ("interval", "calendar"):
            issues.append(f"schedule.mode={self.mode!r} must be 'interval' or 'calendar'")
            return issues
        if self.mode == "interval":
            if self.interval_seconds < 60:
                issues.append(
                    f"schedule.interval_seconds={self.interval_seconds} is below the 60s floor"
                )
        else:
            bad_days = [d for d in self.days if d.lower() not in _WEEKDAY_INDEX]
            if bad_days:
                issues.append(
                    f"schedule.days has unknown value(s) {bad_days}; "
                    f"expected values from {sorted(_WEEKDAY_INDEX)}"
                )
            try:
                _parse_time_of_day(self.time_of_day)
            except ValueError as e:
                issues.append(str(e))
            try:
                _resolve_tz(self.timezone)
            except Exception as e:
                issues.append(f"schedule.timezone={self.timezone!r} is not a known timezone: {e}")
        return issues


def _parse_time_of_day(s: str) -> dt_time:
    try:
        hh, mm = s.split(":")
        h, m = int(hh), int(mm)
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError
        return dt_time(hour=h, minute=m)
    except Exception:
        raise ValueError(f"schedule.time_of_day={s!r} must be 'HH:MM' in 24h format")


def _resolve_tz(name: str):
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)


def compute_next_run(schedule: ScheduleConfig, now: Optional[datetime] = None) -> datetime:
    """Return the next scheduled run time as a UTC-aware datetime.

    interval mode: simply ``now + interval_seconds``. The daemon loop
    tracks its own "last run" separately; this function treats "next" as
    "one interval from whenever it's asked," which is exactly what a
    plain interval loop needs.

    calendar mode: the next occurrence of one of ``days`` (or any day if
    ``days`` is empty) at ``time_of_day`` in ``timezone``, strictly after
    ``now``. If today still qualifies (time hasn't passed yet) and today's
    weekday is allowed, returns today's slot.
    """
    now = now or datetime.now(timezone.utc)
    if schedule.mode == "interval":
        return now + timedelta(seconds=schedule.interval_seconds)

    tz = _resolve_tz(schedule.timezone)
    local_now = now.astimezone(tz)
    tod = _parse_time_of_day(schedule.time_of_day)
    days = {d.lower() for d in schedule.days} or set(_WEEKDAYS)

    candidate = local_now.replace(hour=tod.hour, minute=tod.minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)

    for _ in range(8):  # at most 7 days forward + the starting day
        wd = _WEEKDAYS[candidate.weekday()]
        if wd in days:
            return candidate.astimezone(timezone.utc)
        candidate += timedelta(days=1)
    raise AssertionError("unreachable -- no matching weekday found within 8 days")


def seconds_until(schedule: ScheduleConfig, now: Optional[datetime] = None) -> float:
    """Convenience for the daemon loop: seconds to sleep until the next run."""
    now = now or datetime.now(timezone.utc)
    delta = compute_next_run(schedule, now) - now
    return max(0.0, delta.total_seconds())
