import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import levels as sr  # noqa: E402

PLAN = dict(atr_mult=1.5, rr=2.0, min_rr=1.5, max_stop_atr=2.5)


def bars_from_closes(closes, atr=1.0):
    idx = pd.date_range("2026-10-01 09:30", periods=len(closes), freq="5min", tz="America/New_York")
    closes = np.asarray(closes, float)
    return pd.DataFrame({
        "open": closes, "high": closes + 0.1, "low": closes - 0.1, "close": closes,
        "volume": 1000.0, "atr": atr,
    }, index=idx)


def test_fibonacci_detects_bounce_in_golden_zone():
    # Impulse 100 -> 110, pullback to ~104 (0.6 retracement), then a bounce.
    up = np.linspace(100, 110, 20)
    down = np.linspace(110, 104, 8)
    bounce = [105.0, 106.0]
    df = bars_from_closes(np.r_[up, down, bounce])
    fib = sr.fibonacci_columns(df)
    last = fib.iloc[-1]
    assert last["fib_dir"] == 1
    assert bool(last["fib_bull_zone"])
    assert not fib["fib_bear_zone"].any()
    levels = sr.fib_levels(df.join(fib).iloc[-1])
    assert [l.label for l in levels] == ["Fib 0.382", "Fib 0.5", "Fib 0.618", "Fib 0.786"]
    assert all(not l.strong for l in levels)


def test_fibonacci_ignores_too_deep_pullback():
    up = np.linspace(100, 110, 20)
    down = np.linspace(110, 100.5, 10)  # ~0.95 retracement
    df = bars_from_closes(np.r_[up, down, [101.0]])
    assert not sr.fibonacci_columns(df)["fib_bull_zone"].iloc[-1]


def test_plan_moves_stop_behind_support_and_caps_target_at_resistance():
    # ATR stop (98.50) would sit just above the 98.20 support: move it behind it.
    levels = [sr.Level(98.2, "Mínimo de ayer"), sr.Level(103.5, "Máximo del día")]
    plan = sr.plan_trade("LONG", 100.0, 1.0, levels, **PLAN)
    assert plan["stop"] == 98.05
    assert plan["target"] == 103.35         # 2R (103.90) pulled in before the resistance
    assert plan["ok"] and abs(plan["rr"] - 3.35 / 1.95) < 1e-6


def test_plan_keeps_atr_stop_when_already_behind_support():
    levels = [sr.Level(98.8, "Mínimo de ayer"), sr.Level(104.0, "Máximo del día")]
    plan = sr.plan_trade("LONG", 100.0, 1.0, levels, **PLAN)
    assert plan["stop"] == 98.5
    assert plan["target"] == 103.0          # full 2R fits below the resistance
    assert plan["ok"] and plan["rr"] == 2.0


def test_plan_rejects_trade_right_under_resistance():
    plan = sr.plan_trade("LONG", 100.0, 1.0, [sr.Level(100.8, "Máximo de ayer")], **PLAN)
    assert not plan["ok"]
    assert "Máximo de ayer" in plan["reason"]


def test_plan_short_is_mirror_of_long():
    levels = [sr.Level(101.8, "Máximo de ayer"), sr.Level(96.5, "Mínimo del día")]
    plan = sr.plan_trade("SHORT", 100.0, 1.0, levels, **PLAN)
    assert plan["stop"] == 101.95           # above the resistance
    assert plan["target"] == 96.65          # before the support
    assert plan["ok"] and abs(plan["rr"] - 3.35 / 1.95) < 1e-6


def test_key_levels_include_previous_day_and_session_levels():
    day1 = pd.date_range("2026-10-01 09:30", "2026-10-01 15:55", freq="5min", tz="America/New_York")
    day2 = pd.date_range("2026-10-02 09:30", "2026-10-02 11:00", freq="5min", tz="America/New_York")
    idx = day1.append(day2)
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    df = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2,
                       "close": close, "volume": 1000.0, "atr": 0.5}, index=idx)
    df = df.join(sr.fibonacci_columns(df))
    labels = " ".join(l.label for l in sr.key_levels(df))
    for name in ("Máximo de ayer", "Mínimo de ayer", "Máx. apertura", "Máximo del día"):
        assert name in labels
    prev = df.loc["2026-10-01"]
    assert any(abs(l.price - prev["high"].max()) < 1e-9 for l in sr.key_levels(df))
