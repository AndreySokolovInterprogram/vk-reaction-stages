"""Tests and example for collection/schedule.py.  Run:  python collection/test_schedule.py  (or pytest)."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule import TARGETS_H, plan_table, target_plan, tolerance  # noqa: E402

PUB = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


def test_tolerance():
    # max(20 min, 10% of the mark)
    expected_min = {1: 20, 3: 20, 6: 36, 12: 72, 24: 144, 48: 288, 72: 432}
    assert {t: int(tolerance(t).total_seconds() // 60) for t in TARGETS_H} == expected_min


def test_first_mark_for_a_new_post():
    idx, due, skipped = target_plan(PUB, PUB + timedelta(minutes=5))
    assert (idx, due, skipped) == (0, PUB + timedelta(hours=1), 0)


def test_mark_within_tolerance_is_kept():
    # 1-h mark: deadline 1 h 20 min
    assert target_plan(PUB, PUB + timedelta(minutes=80))[0] == 0
    idx, due, skipped = target_plan(PUB, PUB + timedelta(minutes=81))
    assert (idx, skipped) == (1, 1) and due == PUB + timedelta(hours=3)


def test_late_discovery_skips_marks():
    # first seen at 7 h: marks 1, 3 and 6 (deadline 6 h 36 min) are missed, next is 12 h
    idx, due, skipped = target_plan(PUB, PUB + timedelta(hours=7))
    assert (TARGETS_H[idx], skipped) == (12, 3)


def test_no_marks_left():
    assert target_plan(PUB, PUB + timedelta(hours=72) + timedelta(minutes=433)) == (None, None, 7)


def test_plan_table():
    rows = plan_table(PUB)
    assert [r[0] for r in rows] == list(TARGETS_H)
    assert rows[-1][2] == PUB + timedelta(hours=72, minutes=432)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print(f"\nexample: post published at {PUB:%Y-%m-%d %H:%M} UTC")
    print(f"{'mark, h':>8}  {'due (UTC)':>16}  {'deadline (UTC)':>16}  tolerance")
    for t, due, deadline, tol in plan_table(PUB):
        print(f"{t:>8}  {due:%Y-%m-%d %H:%M}  {deadline:%Y-%m-%d %H:%M}  {int(tol.total_seconds() // 60)} min")
