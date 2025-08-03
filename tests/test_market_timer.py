"""Tests for the market timer utilities."""

import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution import market_timer


class DummyClock:
    def __init__(self, timestamp, next_close):
        self.timestamp = timestamp
        self.next_close = next_close


class DummyApi:
    def __init__(self, clock):
        self._clock = clock

    def get_clock(self):
        return self._clock


def test_should_exit_positions():
    now = datetime.utcnow()

    clock_far = DummyClock(now, now + timedelta(minutes=6))
    api_far = DummyApi(clock_far)
    assert market_timer.should_exit_positions(api_far) is False

    clock_near = DummyClock(now, now + timedelta(minutes=2))
    api_near = DummyApi(clock_near)
    assert market_timer.should_exit_positions(api_near) is True
