import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ALPACA_KEY", "test")
os.environ.setdefault("ALPACA_SECRET", "test")
import bot  # noqa: E402


def synthetic_bars(days=8, drift=0.02, seed=1):
    rng = np.random.default_rng(seed)
    idx = []
    for day in pd.bdate_range("2026-09-21", periods=days):
        idx.extend(pd.date_range(f"{day.date()} 09:30", f"{day.date()} 15:55", freq="5min", tz=bot.NY))
    idx = pd.DatetimeIndex(idx)
    close = 180 + np.cumsum(rng.normal(drift, 0.4, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.4, len(idx))
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.4, len(idx))
    volume = rng.integers(20_000, 120_000, len(idx)).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_indicators_have_expected_columns_and_no_lookahead():
    frame = synthetic_bars()
    full = bot.add_indicators(frame)
    for col in ("ema_fast", "vwap", "rsi", "atr", "adx", "macd_hist", "htf_trend", "market_bias"):
        assert col in full
    # Values for earlier bars must not change when future bars are added.
    cut = bot.add_indicators(frame.iloc[:300])
    cols = ["rsi", "atr", "adx", "vwap", "htf_trend", "market_bias", "swing_high"]
    pd.testing.assert_frame_equal(full[cols].iloc[:300], cut[cols])


def test_analyze_produces_signals_with_full_checklist():
    signals = []
    for seed in range(6):
        for drift in (0.03, -0.03):
            enriched = bot.add_indicators(synthetic_bars(drift=drift, seed=seed))
            for _, row in enriched.iloc[100:].iterrows():
                side, score, checks = bot.analyze(row)
                if side:
                    signals.append((side, score, checks))
    assert signals, "the strategy never fired on trending synthetic data"
    sides = {s for s, _, _ in signals}
    assert sides == {"LONG", "SHORT"}
    for side, score, checks in signals:
        assert score >= bot.MIN_SCORE
        assert len(checks) == 12
        assert all(ok for _, ok in checks[:4])  # mandatory trend filters


def test_levels_and_size():
    stop, tp = bot.compute_levels("LONG", 100.0, 2.0)
    assert (stop, tp) == (98.0, 104.0)
    stop, tp = bot.compute_levels("SHORT", 100.0, 2.0)
    assert (stop, tp) == (102.0, 96.0)
    # 0.5% of 100k = $500 risk / $2 = 250 shares, but notional cap 20% = $20k / $100 = 200.
    qty, risk = bot.position_size(100_000, 200_000, 100.0, 2.0)
    assert (qty, risk) == (200, 500.0)
    qty, _ = bot.position_size(1_000, 1_000, 100.0, 2.0)
    assert qty == 2  # 20% of 1000 / 100


class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.ok = status < 400
        self.text = "x"

    def json(self):
        return self._data

    def raise_for_status(self):
        if not self.ok:
            raise AssertionError(f"HTTP {self.status_code}")


def test_main_end_to_end_places_bracket_after_approval(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # Find a synthetic bar that produces a signal and pretend "now" is just after it.
    for seed in range(20):
        frame = synthetic_bars(drift=0.03, seed=seed)
        enriched = bot.add_indicators(frame)
        hits = [
            t for t, r in enriched.iloc[100:].iterrows()
            if bot.analyze(r)[0] and bot.ENTRY_START <= t.time() < bot.ENTRY_END
        ]
        if hits:
            break
    signal_time = hits[0]
    frame = frame[frame.index <= signal_time]
    now = signal_time + pd.Timedelta(minutes=6)
    expected_side, _, _ = bot.analyze(enriched.loc[signal_time])
    price = float(frame["close"].iloc[-1])

    real_now = pd.Timestamp.now
    monkeypatch.setattr(bot.pd.Timestamp, "now", staticmethod(lambda tz=None: now.tz_convert(tz) if tz else now))
    monkeypatch.setattr(bot, "get_bars", lambda: frame)
    monkeypatch.setattr(bot, "ask_approval", lambda text: "approve")
    monkeypatch.setattr(bot, "time", type("T", (), {"sleep": staticmethod(lambda s: None),
                                                    "monotonic": staticmethod(lambda: 0)}))
    messages, orders = [], []
    monkeypatch.setattr(bot, "telegram", messages.append)

    def fake_request(method, url, headers=None, params=None, json=None, timeout=None):
        path = url.replace(bot.TRADING_URL, "")
        if path == "/v2/clock":
            return FakeResponse({"is_open": True})
        if path == "/v2/account":
            return FakeResponse({"equity": "100000", "buying_power": "200000"})
        if path in ("/v2/positions",) or (path == "/v2/orders" and method == "GET"):
            return FakeResponse([])
        if path == "/v2/orders" and method == "POST":
            orders.append(json)
            return FakeResponse({"id": "abc", "status": "accepted"})
        if path == "/v2/orders/abc":
            return FakeResponse({"id": "abc", "status": "filled", "filled_avg_price": str(price)})
        raise AssertionError(path)

    def fake_get(url, headers=None, params=None, timeout=None):
        assert url.endswith("/trades/latest")
        return FakeResponse({"trade": {"p": price}})

    monkeypatch.setattr(bot.SESSION, "request", fake_request)
    monkeypatch.setattr(bot.SESSION, "get", fake_get)

    bot.main()
    monkeypatch.setattr(bot.pd.Timestamp, "now", real_now)

    assert len(orders) == 1, messages
    order = orders[0]
    assert order["order_class"] == "bracket"
    assert order["side"] == ("buy" if expected_side == "LONG" else "sell")
    stop = float(order["stop_loss"]["stop_price"])
    tp = float(order["take_profit"]["limit_price"])
    if expected_side == "LONG":
        assert stop < price < tp
    else:
        assert tp < price < stop
    assert abs(abs(tp - price) - bot.RR * abs(price - stop)) < 0.02
    assert "Orden enviada" in messages[-1]
    state = bot.load_state()
    assert state["count"] == 1 and state["last_bar"]
