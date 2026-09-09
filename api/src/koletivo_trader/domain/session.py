from __future__ import annotations

from datetime import date, datetime, time, timedelta


GOLD_WINDOWS = (("09:15", "11:00"), ("14:30", "17:00"))
LIVE_SIGNAL_PAD_MINUTES = 5


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")[:2]
    return time(int(hour), int(minute))


def _shift_time(day: date, clock: time, minutes: int) -> time:
    return (datetime.combine(day, clock) + timedelta(minutes=minutes)).time()


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

    def allows(self, ts: datetime, *, pad_minutes: int = 0) -> bool:
        clock = ts.time()
        day = ts.date()
        start = _shift_time(day, self.start, -pad_minutes)
        end = _shift_time(day, self.end, pad_minutes)
        if not (start <= clock <= end):
            return False
        if self.skip_lunch:
            lunch_a = _shift_time(day, self.lunch_start, pad_minutes)
            lunch_b = _shift_time(day, self.lunch_end, -pad_minutes)
            in_lunch = lunch_a < clock < lunch_b if pad_minutes else self.lunch_start <= clock < self.lunch_end
            if in_lunch:
                return False
        if self.gold_only and not any(
            _shift_time(day, a, -pad_minutes) <= clock <= _shift_time(day, b, pad_minutes) for a, b in self.gold
        ):
            return False
        return True

    def allows_live(self, ts: datetime) -> bool:
        """Ao vivo: ouro ± 5 min. Treino e replay continuam no `allows()` exato."""
        return self.allows(ts, pad_minutes=LIVE_SIGNAL_PAD_MINUTES)

    def flatten_day(self, ts: datetime) -> bool:
        return ts.time() >= self.end

    def minutes_from_open(self, ts: datetime) -> float:
        open_at = datetime.combine(ts.date(), self.start)
        return (ts - open_at).total_seconds() / 60.0
