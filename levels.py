"""
Support/resistance levels, Fibonacci retracements and trade planning.

Everything here only looks at the bars it is given, so callers must pass data that
ends at the bar being evaluated (no look-ahead).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

FIB_LOOKBACK = 60          # 5-min bars (~5 h) used to find the latest impulse leg
FIB_MIN_RANGE_ATR = 2.0    # ignore legs smaller than this many ATRs
FIB_RECENT_BARS = 8        # the pullback must have touched the zone this recently
FIB_RATIOS = (0.382, 0.5, 0.618, 0.786)
PIVOT_SPAN = 10            # a major pivot is the extreme of 10 bars on each side
PIVOT_DAYS = 3             # sessions scanned for repeated pivots (S/R zones)
ZONE_MIN_TOUCHES = 3
CLUSTER_ATR = 0.35         # pivots within this many ATRs form one S/R zone
MERGE_ATR = 0.1            # levels closer than this are merged into one
LEVEL_BUFFER_ATR = 0.15    # stops/targets sit this far beyond/before a level
TOUCH_ATR = 0.25           # how close price must come to count as testing a level


@dataclass(frozen=True)
class Level:
    price: float
    label: str
    strong: bool = True


def fibonacci_columns(df):
    """
    For every bar, find the latest impulse leg inside FIB_LOOKBACK bars and flag a
    pullback into the golden zone (0.5-0.786) that is holding above 0.618 (bull) or
    below 0.618 (bear). Only bars up to and including the current one are used.
    """
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    fib_dir = np.zeros(n, dtype=int)
    fib_hi = np.full(n, np.nan)
    fib_lo = np.full(n, np.nan)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(atr[i]):
            continue
        start = max(0, i - FIB_LOOKBACK + 1)
        hi_idx = start + int(np.argmax(high[start:i + 1]))
        lo_idx = start + int(np.argmin(low[start:i + 1]))
        leg = high[hi_idx] - low[lo_idx]
        if leg < FIB_MIN_RANGE_ATR * atr[i]:
            continue

        if lo_idx < hi_idx <= i - 2:
            fib_dir[i], fib_hi[i], fib_lo[i] = 1, high[hi_idx], low[lo_idx]
            pull_idx = hi_idx + 1 + int(np.argmin(low[hi_idx + 1:i + 1]))
            pull = low[pull_idx]
            f50, f618, f786 = (high[hi_idx] - r * leg for r in (0.5, 0.618, 0.786))
            bull[i] = (
                f786 <= pull <= f50
                and f618 < close[i] < high[hi_idx]
                and i - pull_idx < FIB_RECENT_BARS
            )
        elif hi_idx < lo_idx <= i - 2:
            fib_dir[i], fib_hi[i], fib_lo[i] = -1, high[hi_idx], low[lo_idx]
            pull_idx = lo_idx + 1 + int(np.argmax(high[lo_idx + 1:i + 1]))
            pull = high[pull_idx]
            f50, f618, f786 = (low[lo_idx] + r * leg for r in (0.5, 0.618, 0.786))
            bear[i] = (
                f50 <= pull <= f786
                and low[lo_idx] < close[i] < f618
                and i - pull_idx < FIB_RECENT_BARS
            )

    return pd.DataFrame(
        {
            "fib_dir": fib_dir,
            "fib_hi": fib_hi,
            "fib_lo": fib_lo,
            "fib_bull_zone": bull,
            "fib_bear_zone": bear,
        },
        index=df.index,
    )


def fib_levels(row):
    """Retracement levels of the leg active at this bar (weak levels)."""
    if row["fib_dir"] == 0 or pd.isna(row["fib_hi"]):
        return []
    hi, lo = float(row["fib_hi"]), float(row["fib_lo"])
    leg = hi - lo
    if row["fib_dir"] == 1:
        return [Level(hi - r * leg, f"Fib {r:g}", strong=False) for r in FIB_RATIOS]
    return [Level(lo + r * leg, f"Fib {r:g}", strong=False) for r in FIB_RATIOS]


def _major_pivots(df):
    span = 2 * PIVOT_SPAN + 1
    highs = df["high"]
    lows = df["low"]
    # A centered window needs PIVOT_SPAN future bars, so the newest bars are never
    # pivots yet: min_periods=span leaves them NaN.
    is_high = highs.eq(highs.rolling(span, center=True, min_periods=span).max())
    is_low = lows.eq(lows.rolling(span, center=True, min_periods=span).min())
    return sorted(list(highs[is_high]) + list(lows[is_low]))


def _cluster(prices, width):
    zones = []
    for price in prices:
        if zones and price - zones[-1][0] <= width:
            zones[-1].append(price)
        else:
            zones.append([price])
    return zones


def _merge(levels, distance):
    merged = []
    for level in sorted(levels, key=lambda l: l.price):
        if merged and level.price - merged[-1].price <= distance:
            prev = merged[-1]
            labels = prev.label if level.label in prev.label else f"{prev.label}/{level.label}"
            keep = prev if prev.strong or not level.strong else level
            merged[-1] = Level(keep.price, labels, prev.strong or level.strong)
        else:
            merged.append(level)
    return merged


def key_levels(df):
    """S/R levels known at the close of the last bar of df (an enriched frame)."""
    atr = float(df["atr"].iloc[-1])
    dates = np.array(df.index.date)
    today = dates[-1]
    levels = []

    previous_days = sorted(set(dates[dates < today]))
    if previous_days:
        prev = df[dates == previous_days[-1]]
        levels += [
            Level(float(prev["high"].max()), "Máximo de ayer"),
            Level(float(prev["low"].min()), "Mínimo de ayer"),
            Level(float(prev["close"].iloc[-1]), "Cierre de ayer", strong=False),
        ]

    session = df[dates == today]
    if len(session) > 3:
        opening = session.iloc[:3]
        levels += [
            Level(float(opening["high"].max()), "Máx. apertura"),
            Level(float(opening["low"].min()), "Mín. apertura"),
        ]
    # Leave out the last 3 bars so the bar that is moving right now is not
    # counted as an established high/low of the day.
    established = session.iloc[:-3]
    if len(established):
        levels += [
            Level(float(established["high"].max()), "Máximo del día"),
            Level(float(established["low"].min()), "Mínimo del día"),
        ]

    recent_days = sorted(set(dates))[-PIVOT_DAYS:]
    recent = df[np.isin(dates, recent_days)]
    for zone in _cluster(_major_pivots(recent), CLUSTER_ATR * atr):
        if len(zone) >= ZONE_MIN_TOUCHES:
            levels.append(Level(float(np.mean(zone)), f"Zona S/R ({len(zone)} toques)"))

    levels += fib_levels(df.iloc[-1])
    return _merge(levels, MERGE_ATR * atr)


def level_reaction(df, levels, side):
    """
    Describes a bounce off a support (LONG) / resistance (SHORT) or a fresh break of
    a strong level in the trade direction during the last 3-4 bars. Empty if none.
    """
    if len(df) < 4:
        return ""
    atr = float(df["atr"].iloc[-1])
    close = float(df["close"].iloc[-1])
    close_before = float(df["close"].iloc[-4])
    recent = df.iloc[-4:]
    usable = [l for l in levels if l.strong or l.label in ("Fib 0.5", "Fib 0.618")]

    if side == "LONG":
        lowest = float(recent["low"].min())
        for level in sorted((l for l in usable if l.price < close), key=lambda l: -l.price):
            if level.price - 2 * TOUCH_ATR * atr <= lowest <= level.price + TOUCH_ATR * atr:
                return f"Rebote en {level.label} ${level.price:.2f}"
            if level.strong and close_before < level.price:
                return f"Ruptura de {level.label} ${level.price:.2f}"
    else:
        highest = float(recent["high"].max())
        for level in sorted((l for l in usable if l.price > close), key=lambda l: l.price):
            if level.price - TOUCH_ATR * atr <= highest <= level.price + 2 * TOUCH_ATR * atr:
                return f"Rechazo en {level.label} ${level.price:.2f}"
            if level.strong and close_before > level.price:
                return f"Ruptura de {level.label} ${level.price:.2f}"
    return ""


def plan_trade(side, entry, atr, levels, *, atr_mult, rr, min_rr, max_stop_atr):
    """
    Stop: ATR-based, moved behind the nearest strong support/resistance when that
    level sits inside the ATR stop (but never wider than max_stop_atr).
    Target: rr times the risk, pulled in before the next strong level in the way.
    Returns a dict; ok=False with a reason when the room to the next level is too small.
    """
    buffer = LEVEL_BUFFER_ATR * atr
    strong = [l for l in levels if l.strong]
    up = side == "LONG"
    sign = 1 if up else -1

    stop = entry - sign * atr_mult * atr
    stop_note = f"{atr_mult:g}×ATR"
    widest = entry - sign * max_stop_atr * atr
    if up:
        guards = [l for l in strong if widest + buffer <= l.price <= entry - 0.2 * atr]
        guard = max(guards, key=lambda l: l.price, default=None)
        if guard and guard.price - buffer < stop:
            stop, stop_note = guard.price - buffer, f"debajo de {guard.label} ${guard.price:.2f}"
    else:
        guards = [l for l in strong if entry + 0.2 * atr <= l.price <= widest - buffer]
        guard = min(guards, key=lambda l: l.price, default=None)
        if guard and guard.price + buffer > stop:
            stop, stop_note = guard.price + buffer, f"encima de {guard.label} ${guard.price:.2f}"

    stop = round(stop, 2)
    risk = abs(entry - stop)
    target = round(entry + sign * rr * risk, 2)
    target_note = f"{rr:g}R"

    ahead = [l for l in strong if (l.price > entry if up else l.price < entry)]
    blocker = min(ahead, key=lambda l: abs(l.price - entry), default=None)
    if blocker and (blocker.price - buffer < target if up else blocker.price + buffer > target):
        target = round(blocker.price - sign * buffer, 2)
        target_note = f"antes de {blocker.label} ${blocker.price:.2f}"

    reward = sign * (target - entry)
    real_rr = reward / risk if risk > 0 else 0.0
    plan = {
        "ok": real_rr >= min_rr,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "rr": real_rr,
        "stop_note": stop_note,
        "target_note": target_note,
        "reason": "",
    }
    if not plan["ok"]:
        where = "Resistencia" if up else "Soporte"
        plan["reason"] = (
            f"{where} {blocker.label} en ${blocker.price:.2f} demasiado cerca "
            f"(R:R 1:{real_rr:.1f}, mínimo 1:{min_rr:g})"
            if blocker else f"R:R 1:{real_rr:.1f} insuficiente"
        )
    return plan


def nearest(levels, price, side, count=3):
    """Closest levels above (side='up') or below (side='down') price."""
    if side == "up":
        found = sorted((l for l in levels if l.price > price), key=lambda l: l.price)
    else:
        found = sorted((l for l in levels if l.price < price), key=lambda l: -l.price)
    return found[:count]
