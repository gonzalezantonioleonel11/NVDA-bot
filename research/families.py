"""Intraday strategy families for the research run. Each yields (variant name, trades DataFrame).

Signals use 5-minute information; entries and exits are simulated on 1-minute bars.
"""
import itertools

import numpy as np
import pandas as pd

import bot
from research.engine import MIN_STOP, close_trigger, exit_trade, record

OPEN = 9 * 60 + 30


def _opening_range(m, minutes):
    """Per day: (first, end) minute indices and the range's open/high/low/close/volume, or None."""
    out = []
    for di in range(len(m.days)):
        w = m.window(di, minutes)
        if w is None:
            out.append(None)
            continue
        a, b = w
        out.append((a, b, m.O[a], m.H[a:b].max(), m.L[a:b].min(), m.C[b - 1], m.V[a:b].sum()))
    vols = np.array([r[6] if r else np.nan for r in out])
    rvol = vols / pd.Series(vols).rolling(14, min_periods=10).mean().shift(1).to_numpy()
    return out, rvol


def _top_per_day(cands, n):
    """Keep the n candidates with the highest relative volume each day."""
    cands.sort(key=lambda c: (c[0], -c[1]))
    out, last_day, count = [], None, 0
    for c in cands:
        if c[0] != last_day:
            last_day, count = c[0], 0
        if count < n:
            out.append(c)
            count += 1
    return out


# --------------------------------------------------------------------------- ORB 5 minutes
def orb5(markets, costs, top_n=3):
    """5-minute opening range breakout in the direction of the first candle (Zarattini et al. style)."""
    ranges = {s: _opening_range(m, 5) for s, m in markets.items()}
    for entry, stop_mode, tp_r, rv_min in itertools.product(
            ("market", "cierre1m"), ("or", "atr10"), (None, 3.0), (1.0, 1.5)):
        cands = []
        for s, m in markets.items():
            rows, rvol = ranges[s]
            for di, r in enumerate(rows):
                if r and rvol[di] >= rv_min and np.isfinite(m.atr14[di]) and r[5] != r[2]:
                    cands.append((m.days[di], rvol[di], s, di))
        trades = []
        for _, rank, s, di in _top_per_day(cands, top_n):
            m = markets[s]
            a, b, o0, h0, l0, c0, _ = ranges[s][0][di]
            e = m.flat_i[di]
            side = 1 if c0 > o0 else -1
            if entry == "market":
                j, fill, intrabar = b, m.O[b] * (1 + side * costs.entry_bps / 1e4), False
            else:
                got = close_trigger(m, b, m.minute_index(di, 11 * 60),
                                    h0 if side == 1 else None, l0 if side == -1 else None, costs)
                if got is None:
                    continue
                j, side, fill, intrabar = got
            stop = (l0 if side == 1 else h0) if stop_mode == "or" else fill - side * 0.10 * m.atr14[di]
            dist = side * (fill - stop)
            if dist < MIN_STOP * fill or j >= e:
                continue
            tp = fill + side * tp_r * dist if tp_r else None
            jo, px, why = exit_trade(m, j, e, side, fill, stop, tp, intrabar, costs)
            trades.append(record(m, di, side, fill, dist, j, jo, px, why, rank))
        yield f"ORB5 entrada={entry} stop={stop_mode} tp={tp_r or 'cierre'} rvol>={rv_min}", pd.DataFrame(trades)


# --------------------------------------------------------------------------- ORB 15 / 30 minutes
def orb_range(markets, costs, top_n=3, confirm=1):
    """Classic opening range breakout on both sides, confirmed by `confirm` consecutive 1-minute
    closes outside the range, until 12:00."""
    for minutes in (15, 30):
        ranges = {s: _opening_range(m, minutes) for s, m in markets.items()}
        for stop_mode, tp_r, rv_min in itertools.product(("opp", "mid"), (None, 2.0), (0.0, 1.2)):
            cands = []
            for s, m in markets.items():
                rows, rvol = ranges[s]
                for di, r in enumerate(rows):
                    if r and np.isfinite(rvol[di]) and rvol[di] >= rv_min:
                        cands.append((m.days[di], rvol[di], s, di))
            trades = []
            for _, rank, s, di in _top_per_day(cands, top_n):
                m = markets[s]
                a, b, o0, hi, lo, c0, _ = ranges[s][0][di]
                e = m.flat_i[di]
                got = close_trigger(m, b, m.minute_index(di, 12 * 60), hi, lo, costs, confirm)
                if got is None:
                    continue
                j, side, fill, intrabar = got
                stop = (lo if side == 1 else hi) if stop_mode == "opp" else (hi + lo) / 2
                dist = side * (fill - stop)
                if dist < MIN_STOP * fill or j >= e:
                    continue
                tp = fill + side * tp_r * dist if tp_r else None
                jo, px, why = exit_trade(m, j, e, side, fill, stop, tp, intrabar, costs)
                trades.append(record(m, di, side, fill, dist, j, jo, px, why, rank))
            yield (f"ORB{minutes} stop={stop_mode} tp={tp_r or 'cierre'} rvol>={rv_min or 'todos'}",
                   pd.DataFrame(trades))


# --------------------------------------------------------------------------- Noise-area intraday momentum
def _noise_inputs(m, every):
    """Check minutes (bar closing at 10:00, 10:30... or 10:00, 11:00...), sigma per check, VWAP."""
    checks = [OPEN + 30 + k * every - 1 for k in range(6 * 60 // every)]
    checks = [c for c in checks if c <= 15 * 60 + 29]
    tp = (m.H + m.L + m.C) / 3
    vwap = np.empty_like(tp)
    for a, b in zip(m.day_first, m.day_last):
        vwap[a:b + 1] = np.cumsum(tp[a:b + 1] * m.V[a:b + 1]) / np.maximum(np.cumsum(m.V[a:b + 1]), 1e-9)
    nd = len(m.days)
    bar = np.full((nd, len(checks)), -1)
    for di in range(nd):
        a, b = m.day_first[di], m.day_last[di]
        mods = m.mod[a:b + 1]
        k = np.searchsorted(mods, checks)
        valid = (k < len(mods)) & (mods[np.minimum(k, len(mods) - 1)] == np.array(checks))
        bar[di, valid] = a + k[valid]
    move = np.where(bar >= 0, np.abs(m.C[np.maximum(bar, 0)] / m.day_open[:, None] - 1), np.nan)
    sigma = pd.DataFrame(move).rolling(14, min_periods=10).mean().shift(1).to_numpy()
    prev_close = np.r_[np.nan, m.day_close[:-1]]
    return bar, sigma, prev_close, vwap


def noise_momentum(markets, costs):
    """Long above / short below the 'noise area' around the open, trailing exit at band or VWAP
    (Zarattini, Aziz & Barbon 2024, simplified: fixed-fractional sizing instead of volatility targeting)."""
    inputs = {}
    for universe, every, mult, trail in itertools.product(("SPY+QQQ", "todas"), (30, 60), (1.0, 1.5),
                                                          ("banda+vwap", "vwap")):
        trades = []
        for s, m in markets.items():
            if universe != "todas" and s not in ("SPY", "QQQ"):
                continue
            if (s, every) not in inputs:
                inputs[(s, every)] = _noise_inputs(m, every)
            bar, sigma, prev_close, vwap = inputs[(s, every)]
            for di in range(len(m.days)):
                if not np.isfinite(prev_close[di]) or not np.isfinite(m.atr14[di]):
                    continue
                e = m.flat_i[di]
                ub0, lb0 = max(m.day_open[di], prev_close[di]), min(m.day_open[di], prev_close[di])
                pos, fill, dist, j_in, n_today = 0, 0.0, 0.0, 0, 0
                for ci in range(bar.shape[1]):
                    b = bar[di, ci]
                    if b < 0 or not np.isfinite(sigma[di, ci]) or b + 1 >= e:
                        continue
                    ub, lb = ub0 * (1 + mult * sigma[di, ci]), lb0 * (1 - mult * sigma[di, ci])
                    price = m.C[b]
                    if pos == 0 and n_today < 3:
                        side = 1 if price > ub else (-1 if price < lb else 0)
                        if side:
                            band = ub if side == 1 else lb
                            level = vwap[b] if trail == "vwap" else (max(band, vwap[b]) if side == 1 else min(band, vwap[b]))
                            if side * (price - level) <= 0:
                                level = band
                            j_in = b + 1
                            fill = m.O[j_in] * (1 + side * costs.entry_bps / 1e4)
                            # Risk unit: distance to the trailing level, at least 10% of the daily ATR.
                            dist = max(side * (fill - level), 0.10 * m.atr14[di], MIN_STOP * fill)
                            pos, n_today = side, n_today + 1
                    elif pos != 0:
                        band = ub if pos == 1 else lb
                        level = vwap[b] if trail == "vwap" else (max(band, vwap[b]) if pos == 1 else min(band, vwap[b]))
                        if pos * (price - level) < 0:
                            px = m.O[b + 1] * (1 - pos * costs.exit_bps / 1e4)
                            trades.append(record(m, di, pos, fill, dist, j_in, b + 1, px, "TRAIL", 0.0))
                            pos = 0
                if pos != 0:
                    px = m.O[e] * (1 - pos * costs.exit_bps / 1e4)
                    trades.append(record(m, di, pos, fill, dist, j_in, e, px, "EOD", 0.0))
        yield f"Ruido {universe} cada{every}m banda x{mult} salida={trail}", pd.DataFrame(trades)


# --------------------------------------------------------------------------- the current bot strategy
_FRAMES = {}


def _bot_frame(m):
    key = (m.symbol, len(m.t), int(m.t[0]))
    if key not in _FRAMES:
        b5 = m.b5
        ind = bot.add_indicators(b5[["open", "high", "low", "close", "volume"]])
        ind["i1"] = b5["i1"]
        _FRAMES[key] = ind
    return _FRAMES[key]


def current_strategy(markets, costs):
    """The live NVDA strategy applied to every symbol, with the RSI rule as a variant."""
    for rsi_mode, struct, atr_mult, rr in itertools.product(("cruce", "banda", "sin"), (True, False),
                                                            (1.5, 2.5), (1.5, 2.5)):
        trades = []
        for s, m in markets.items():
            df = _bot_frame(m)
            base = ((df["adx"] > bot.ADX_MIN) & df["vol_ok"] & (df["atr"] > 0) & df["vwap"].notna()
                    & df["rsi"].notna() & df["rsi_prev"].notna())
            L = base & (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["vwap"]) & (df["macd_hist"] > 0)
            S = base & (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["vwap"]) & (df["macd_hist"] < 0)
            if rsi_mode == "cruce":
                L &= (df["rsi_prev"] <= 45) & (df["rsi"] > 45)
                S &= (df["rsi_prev"] >= 55) & (df["rsi"] < 55)
            elif rsi_mode == "banda":
                L &= (df["rsi"] > 50) & (df["rsi"] < 70)
                S &= (df["rsi"] < 50) & (df["rsi"] > 30)
            if struct:
                L &= df["structure_up_recent"] & (df["fvg_bull_recent"] | df["sweep_bull_recent"]) & (df["market_bias"] == 1)
                S &= df["structure_down_recent"] & (df["fvg_bear_recent"] | df["sweep_bear_recent"]) & (df["market_bias"] == -1)
            mod5 = (df.index.hour * 60 + df.index.minute).to_numpy()
            window = (mod5 >= 9 * 60 + 45) & (mod5 + 5 < 15 * 60 + 20)
            sig = np.where(L.to_numpy() & window, 1, np.where(S.to_numpy() & window, -1, 0))
            i1 = df["i1"].to_numpy()
            day_of_min = np.repeat(np.arange(len(m.days)), m.day_last - m.day_first + 1)
            close5, atr = df["close"].to_numpy(), df["atr"].to_numpy()
            busy, count = -1, {}
            for i in np.flatnonzero(sig[:-1]):
                k = i1[i + 1]  # first minute of the next 5-minute bar
                di = day_of_min[k]
                if day_of_min[i1[i]] != di:
                    continue
                e = m.flat_i[di]
                if k <= busy or k >= e or count.get(di, 0) >= bot.MAX_TRADES_DAY:
                    continue
                side = int(sig[i])
                fill = m.O[k] * (1 + side * costs.entry_bps / 1e4)
                stop = close5[i] - side * atr[i] * atr_mult
                tp = close5[i] + side * atr[i] * atr_mult * rr
                dist = side * (fill - stop)
                if dist < MIN_STOP * fill:
                    continue
                jo, px, why = exit_trade(m, k, e, side, fill, stop, tp, False, costs)
                trades.append(record(m, di, side, fill, dist, k, jo, px, why, 0.0))
                busy, count[di] = jo, count.get(di, 0) + 1
        yield f"Actual RSI={rsi_mode} estructura={'si' if struct else 'no'} atr{atr_mult} rr{rr}", pd.DataFrame(trades)


FAMILIES = {
    "ORB5": orb5,
    "ORB15/30": orb_range,
    "ORB15/30-2velas": lambda markets, costs: orb_range(markets, costs, confirm=2),
    "Ruido": noise_momentum,
    "Actual": current_strategy,
}
