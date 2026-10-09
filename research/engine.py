"""Trade simulation on 5-minute bars and portfolio aggregation for the research runs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_STOP = 0.0005  # stops tighter than 0.05% of price are not traded


@dataclass(frozen=True)
class Costs:
    entry_bps: float = 1.0   # market or stop entry
    stop_bps: float = 3.0    # stop-loss exits slip more in fast moves
    exit_bps: float = 1.0    # market exits (signal or end of day)

    def scaled(self, factor):
        return Costs(self.entry_bps * factor, self.stop_bps * factor, self.exit_bps * factor)


class Market:
    """One symbol's regular session: 1-minute arrays for execution, 5-minute bars for signals."""

    def __init__(self, symbol, df1):
        self.symbol = symbol
        self.O, self.H, self.L, self.C, self.V = (df1[k].to_numpy() for k in ("o", "h", "l", "c", "v"))
        self.t = df1.index.tz_convert("UTC").as_unit("ns").asi8
        self.mod = (df1.index.hour * 60 + df1.index.minute).to_numpy()
        codes, days = pd.factorize(df1.index.date)
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        ends = np.r_[starts[1:], len(codes)] - 1
        self.days, self.day_first, self.day_last = np.asarray(days), starts, ends
        self.year = np.array([d.year for d in self.days])
        last = self.mod[ends]
        # Session close: 16:00, or 13:00 on half days.
        self.close_mod = np.where(last >= 955, 960, np.where((last >= 775) & (last < 785), 780, last + 1))
        # Flatten 10 minutes before the close, like the live bot.
        self.flat_i = np.array([min(a + int(np.searchsorted(self.mod[a:b + 1], cm - 10)), b)
                                for a, b, cm in zip(starts, ends, self.close_mod)])
        high = np.maximum.reduceat(self.H, starts)
        low = np.minimum.reduceat(self.L, starts)
        close = self.C[ends]
        prev_close = np.r_[np.nan, close[:-1]]
        tr = np.nanmax(np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0)
        self.atr14 = pd.Series(tr).rolling(14).mean().shift(1).to_numpy()
        self.day_open = self.O[starts]
        self.day_close = close
        self._b5 = None
        self._df1 = df1

    def minute_index(self, di, minute):
        """First 1-minute index of day di whose start is >= minute (capped at the flatten minute)."""
        a, e = self.day_first[di], self.flat_i[di]
        return min(a + int(np.searchsorted(self.mod[a:e + 1], minute)), e)

    def window(self, di, minutes):
        """[first, end) 1-minute indices of the first `minutes` minutes of the session."""
        a = self.day_first[di]
        if self.mod[a] != 9 * 60 + 30:
            return None
        end = a + int(np.searchsorted(self.mod[a:self.day_last[di] + 1], 9 * 60 + 30 + minutes))
        return (a, end) if end < self.flat_i[di] else None

    @property
    def b5(self):
        """5-minute bars built from the 1-minute data, with the first 1-minute index of each bar."""
        if self._b5 is None:
            df = self._df1.assign(i=np.arange(len(self._df1)))
            g = df.resample("5min", label="left", closed="left")
            b = pd.DataFrame({"open": g["o"].first(), "high": g["h"].max(), "low": g["l"].min(),
                              "close": g["c"].last(), "volume": g["v"].sum(), "i1": g["i"].first()}).dropna()
            b["i1"] = b["i1"].astype(np.int64)
            self._b5 = b
        return self._b5


def _stopped_in_entry_bar(m, k, side, stop):
    """Whether the stop is hit after a pending stop order fills inside bar k.

    Uses the usual OHLC path convention (bullish bar: open -> low -> high -> close; bearish bar:
    open -> high -> low -> close) and only looks at the part of the path after the fill. A take
    profit is never credited inside the entry bar, which keeps the estimate conservative.
    """
    bullish = m.C[k] >= m.O[k]
    if side == 1:
        return (m.C[k] if bullish else m.L[k]) <= stop
    return (m.H[k] if bullish else m.C[k]) >= stop


def exit_trade(m, k, e, side, fill, stop, tp, intrabar, costs):
    """Scan bars k..e-1 (k = entry bar) for stop / take profit; otherwise exit at the open of bar e."""
    e = max(e, k + 1)
    if intrabar:
        if _stopped_in_entry_bar(m, k, side, stop):
            return k, stop * (1 - side * costs.stop_bps / 1e4), "SL"
        k += 1
        if k >= e:
            return e, m.O[e] * (1 - side * costs.exit_bps / 1e4), "EOD"
    if side == 1:
        s_hit = m.L[k:e] <= stop
        t_hit = (m.H[k:e] >= tp) if tp is not None else None
    else:
        s_hit = m.H[k:e] >= stop
        t_hit = (m.L[k:e] <= tp) if tp is not None else None
    fs = int(np.argmax(s_hit)) if s_hit.any() else None
    ft = int(np.argmax(t_hit)) if t_hit is not None and t_hit.any() else None
    if fs is not None and (ft is None or fs <= ft):  # both in one bar: assume the stop (conservative)
        j = k + fs
        px = min(stop, m.O[j]) if side == 1 else max(stop, m.O[j])
        return j, px * (1 - side * costs.stop_bps / 1e4), "SL"
    if ft is not None:
        j = k + ft
        px = max(tp, m.O[j]) if side == 1 else min(tp, m.O[j])
        return j, px, "TP"
    return e, m.O[e] * (1 - side * costs.exit_bps / 1e4), "EOD"


def stop_trigger(m, k0, k1, level_long=None, level_short=None, costs=None):
    """First bar in [k0, k1) where a pending buy/sell stop triggers -> (bar, side, fill, intrabar)."""
    best = None
    if level_long is not None:
        hit = np.flatnonzero(m.H[k0:k1] >= level_long)
        if hit.size:
            j = k0 + int(hit[0])
            gap = m.O[j] >= level_long
            best = (j, 1, max(level_long, m.O[j]) * (1 + costs.entry_bps / 1e4), not gap)
    if level_short is not None:
        hit = np.flatnonzero(m.L[k0:k1] <= level_short)
        if hit.size:
            j = k0 + int(hit[0])
            if best is None or j < best[0]:
                gap = m.O[j] <= level_short
                best = (j, -1, min(level_short, m.O[j]) * (1 - costs.entry_bps / 1e4), not gap)
    return best


def close_trigger(m, k0, k1, level_long=None, level_short=None, costs=None):
    """Breakout confirmed by a 1-minute close beyond the level, entered at the next minute's open.

    Returns (entry minute, side, fill, intrabar=False) or None. Unlike a resting stop order this
    needs no assumption about the path inside a bar, and the live bot can do exactly the same.
    """
    best = None
    if level_long is not None:
        hit = np.flatnonzero(m.C[k0:k1] > level_long)
        if hit.size:
            best = (k0 + int(hit[0]), 1)
    if level_short is not None:
        hit = np.flatnonzero(m.C[k0:k1] < level_short)
        if hit.size and (best is None or k0 + int(hit[0]) < best[0]):
            best = (k0 + int(hit[0]), -1)
    if best is None:
        return None
    j, side = best[0] + 1, best[1]
    if j >= len(m.O):
        return None
    return j, side, m.O[j] * (1 + side * costs.entry_bps / 1e4), False


def record(m, di, side, fill, stop_dist, j_in, j_out, px_out, why, rank):
    return {
        "sym": m.symbol, "day": m.days[di], "year": int(m.year[di]), "side": side,
        "t_in": int(m.t[j_in]), "t_out": int(m.t[min(j_out, len(m.t) - 1)]),
        "r": side * (px_out - fill) / stop_dist, "why": why, "stop_pct": stop_dist / fill, "rank": rank,
    }


def trade_stats(tr):
    if tr is None or len(tr) == 0:
        return {"n": 0, "R": 0.0, "pf": 0.0, "wr": 0.0, "exp": 0.0, "t": 0.0, "yrs_pos": 0.0, "per_week": 0.0}
    r = tr["r"].to_numpy()
    gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
    by_year = tr.groupby("year")["r"].sum()
    days = pd.to_datetime(pd.Series(tr["day"]))
    weeks = max((days.max() - days.min()).days / 7, 1)
    return {
        "n": len(r), "R": float(r.sum()), "pf": float(gains / losses) if losses else float("inf"),
        "wr": float((r > 0).mean() * 100), "exp": float(r.mean()),
        "t": float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0,
        "yrs_pos": float((by_year > 0).mean() * 100), "per_week": len(r) / weeks,
    }


def portfolio(trades, risk, max_pos=3, lev_cap=4.0):
    """Fixed-fractional sizing with a cap on simultaneous positions and on total leverage.

    Each trade risks `risk` of equity, unless its notional would exceed lev_cap/max_pos of
    equity (tight stops), in which case it is scaled down. Returns (daily returns, accepted).
    """
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float), trades
    tr = trades.sort_values(["t_in", "rank"], ascending=[True, False])
    t_in, t_out = tr["t_in"].to_numpy(), tr["t_out"].to_numpy()
    open_until, keep, frac = [], [], []
    for i in range(len(tr)):
        open_until = [x for x in open_until if x > t_in[i]]
        if len(open_until) >= max_pos:
            continue
        keep.append(i)
        open_until.append(t_out[i])
    acc = tr.iloc[keep].copy()
    acc["f"] = np.minimum(risk, lev_cap / max_pos * acc["stop_pct"].to_numpy())
    acc["ret"] = acc["f"] * acc["r"]
    daily = acc.groupby("day")["ret"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index(), acc


def monthly_stats(daily, start=None, end=None):
    if len(daily) == 0:
        return {}
    idx = pd.bdate_range(start or daily.index.min(), end or daily.index.max())
    d = daily.reindex(idx, fill_value=0.0)
    equity = (1 + d).cumprod()
    months = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    years = len(d) / 252
    dd = (1 - equity / equity.cummax()).max()
    return {
        "months": len(months), "mean_m": months.mean() * 100, "median_m": months.median() * 100,
        "pct_ge3": (months >= 0.03).mean() * 100, "pct_neg": (months < 0).mean() * 100,
        "worst_m": months.min() * 100, "best_m": months.max() * 100,
        "cagr": (equity.iloc[-1] ** (1 / years) - 1) * 100 if years > 0 else 0.0,
        "max_dd": dd * 100, "monthly": months,
    }
