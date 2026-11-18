"""Measurement schedule of the VK panel (standalone excerpt of the collector logic, no database or API code).

Each live post is measured at scheduled ages (hours after publication). A measurement taken at time t counts for
mark T if t <= published + T + tolerance(T), with tolerance(T) = max(20 min, 10% of T). If the deadline of a mark
has already passed when the post is first seen (or when the collector gets to it), the mark is skipped ("missed")
and the next mark is planned. The actual age of every measurement is stored separately.

    python collection/schedule.py            # prints the plan for a post published now
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

TARGETS_H = (1, 3, 6, 12, 24, 48, 72)   # scheduled ages, hours
TOLERANCE_MIN_MINUTES = 20               # lower bound of the tolerance
TOLERANCE_FRACTION = 0.1                 # tolerance as a fraction of the mark


def tolerance(target_h: float, min_minutes: float = TOLERANCE_MIN_MINUTES,
              fraction: float = TOLERANCE_FRACTION) -> timedelta:
    """Tolerance of a scheduled mark: max(min_minutes, fraction * mark)."""
    return max(timedelta(minutes=min_minutes), timedelta(hours=target_h * fraction))


def target_plan(published: datetime, after: datetime, start_index: int = 0, targets=TARGETS_H,
                min_minutes: float = TOLERANCE_MIN_MINUTES, fraction: float = TOLERANCE_FRACTION):
    """First mark (from index start_index) whose deadline has not passed at time `after`.

    Returns (index, due_at, number_of_skipped_marks); index and due_at are None when no mark is left."""
    skipped = 0
    for idx in range(start_index, len(targets)):
        due = published + timedelta(hours=targets[idx])
        if after <= due + tolerance(targets[idx], min_minutes, fraction):
            return idx, due, skipped
        skipped += 1
    return None, None, skipped


def plan_table(published: datetime, targets=TARGETS_H):
    """Rows (mark_h, due_at, deadline, tolerance) for a post published at `published`."""
    rows = []
    for t in targets:
        due = published + timedelta(hours=t)
        tol = tolerance(t)
        rows.append((t, due, due + tol, tol))
    return rows


if __name__ == "__main__":
    pub = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    print(f"post published at {pub:%Y-%m-%d %H:%M} UTC")
    print(f"{'mark, h':>8}  {'due (UTC)':>16}  {'deadline (UTC)':>16}  tolerance")
    for t, due, deadline, tol in plan_table(pub):
        print(f"{t:>8}  {due:%Y-%m-%d %H:%M}  {deadline:%Y-%m-%d %H:%M}  {int(tol.total_seconds() // 60)} min")
