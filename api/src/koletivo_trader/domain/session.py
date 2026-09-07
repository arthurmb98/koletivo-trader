from __future__ import annotations

from datetime import datetime, time


GOLD_WINDOWS = (("09:15", "11:00"), ("14:30", "17:00"))


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")[:2]
    return time(int(hour), int(minute))


class SessionFilter:
    def __init__(
        self,
        session_start: str = "09:15",
        session_end: str = "17:00",
        skip_lunch: bool = True,
        lunch_start: str = "11:00",
        lunch_end: str = "14:30",
        gold_hours_only: bool = True,
    ) -> None:
        self.start = _parse_hhmm(session_start)
        self.end = _parse_hhmm(session_end)
        self.skip_lunch = skip_lunch
        self.lunch_start = _parse_hhmm(lunch_start)
        self.lunch_end = _parse_hhmm(lunch_end)
        self.gold_only = gold_hours_only
        self.gold = [(_parse_hhmm(start), _parse_hhmm(end)) for start, end in GOLD_WINDOWS]

    @classmethod
    def from_config(cls, cfg) -> "SessionFilter":
        flt = cfg.filters
        return cls(
            flt.session_start,
            flt.session_end,
            flt.skip_lunch,
            flt.lunch_start,
            flt.lunch_end,
            flt.gold_hours_only,
        )

    def allows(self, ts: datetime) -> bool:
        clock = ts.time()
        if not (self.start <= clock <= self.end):
            return False
        if self.skip_lunch and self.lunch_start <= clock < self.lunch_end:
            return False
        if self.gold_only and not any(a <= clock <= b for a, b in self.gold):
            return False
        return True

    def flatten_day(self, ts: datetime) -> bool:
        return ts.time() >= self.end

    def minutes_from_open(self, ts: datetime) -> float:
        open_at = datetime.combine(ts.date(), self.start)
        return (ts - open_at).total_seconds() / 60.0
