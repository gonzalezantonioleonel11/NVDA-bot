"""
Backtest de la estrategia de bot.py sobre velas históricas de 5 minutos de Alpaca.

Usa los mismos indicadores (add_indicators) y la misma señal (evaluate) que el bot
en vivo, con el mismo tamaño de posición, stop, take profit, límite diario, horario
de entradas y cierre de fin de día. No envía órdenes: solo lee datos.

Supuestos:
- Aprobás todas las señales.
- La entrada se ejecuta en la apertura de la vela siguiente a la señal.
- Si una misma vela toca el stop y el take profit, se asume el stop (lo conservador).
- Se descuenta un slippage fijo por acción en la entrada y en las salidas a mercado.

Uso:  python backtest.py --months 6 [--feed iex|sip] [--telegram]
"""
import argparse
import csv
import math
import os

import pandas as pd

import bot

FIVE_MIN = pd.Timedelta(minutes=5)
WARMUP_BARS = 100  # the live bot needs at least 100 closed bars before trading


def fetch_bars(start, end, feed):
    url = f"{bot.DATA_URL}/v2/stocks/{bot.SYMBOL}/bars"
    params = {
        "timeframe": "5Min",
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "adjustment": "split",  # keep prices continuous across stock splits
        "feed": feed,
    }
    bars = []
    while True:
        response = bot.SESSION.get(url, headers=bot.HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        bars.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    frame = pd.DataFrame(bars)
    if frame.empty:
        return frame
    frame["t"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(bot.NY)
    frame = frame.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame = frame.set_index("t")[["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.between_time("09:30", "15:59")


def fetch_closes(start, end):
    """Real close time of each session, so half days are simulated like the live bot."""
    days = bot.api("GET", "/v2/calendar", params={
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    })
    return {
        day["date"]: pd.Timestamp(f"{day['date']} {day['close']}").tz_localize(bot.NY)
        for day in days
    }


def exit_trade(times, opens, highs, lows, closes, entry_idx, side, stop, tp, flat_time, slippage):
    sign = 1 if side == "LONG" else -1
    day = times[entry_idx].date()
    last = entry_idx
    for j in range(entry_idx, len(times)):
        if times[j].date() != day:
            break
        last = j
        if times[j] >= flat_time:
            return j, opens[j] - sign * slippage, "CIERRE FIN DE DÍA"
        o, h, l = opens[j], highs[j], lows[j]
        if side == "LONG":
            if o <= stop:
                return j, o - slippage, "STOP LOSS"
            if o >= tp:
                return j, o, "TAKE PROFIT"
            if l <= stop:
                return j, stop - slippage, "STOP LOSS"
            if h >= tp:
                return j, tp, "TAKE PROFIT"
        else:
            if o >= stop:
                return j, o + slippage, "STOP LOSS"
            if o <= tp:
                return j, o, "TAKE PROFIT"
            if h >= stop:
                return j, stop + slippage, "STOP LOSS"
            if l <= tp:
                return j, tp, "TAKE PROFIT"
    # No bar left before the forced close (missing data): exit at the last close.
    return last, closes[last] - sign * slippage, "CIERRE FIN DE DÍA"


def simulate(frame, session_closes, start_equity, slippage):
    df = bot.add_indicators(frame)
    times = list(df.index)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    equity = start_equity
    trades = []
    skipped = {"ya en posición": 0, "límite diario": 0, "menos de 1 acción": 0}
    trades_per_day = {}
    busy_until = -1

    for i in range(WARMUP_BARS, len(df) - 1):
        bar_time = times[i]
        day = bar_time.strftime("%Y-%m-%d")
        close_ny = session_closes.get(day)
        if close_ny is None:
            continue
        # The live bot acts when the bar closes, and only inside the entry window.
        entry_time = bar_time + FIVE_MIN
        if bar_time.time() < bot.ENTRY_START or entry_time >= close_ny - bot.ENTRY_STOP_BEFORE_CLOSE:
            continue

        row = df.iloc[i]
        side = bot.evaluate(row)
        if not side:
            continue
        if i <= busy_until:
            skipped["ya en posición"] += 1
            continue
        if trades_per_day.get(day, 0) >= bot.MAX_TRADES_DAY:
            skipped["límite diario"] += 1
            continue
        entry_idx = i + 1
        if times[entry_idx].strftime("%Y-%m-%d") != day:
            continue

        signal_price = float(row["close"])
        stop_distance = float(row["atr"]) * bot.ATR_MULT
        qty = min(
            math.floor(equity * (bot.RISK_PCT / 100.0) / stop_distance),
            math.floor(equity * (bot.MAX_NOTIONAL_PCT / 100.0) / signal_price),
        )
        if qty < 1:
            skipped["menos de 1 acción"] += 1
            continue

        sign = 1 if side == "LONG" else -1
        stop = round(signal_price - sign * stop_distance, 2)
        tp = round(signal_price + sign * stop_distance * bot.RR, 2)
        entry = opens[entry_idx] + sign * slippage
        flat_time = close_ny - bot.FORCE_FLAT_BEFORE_CLOSE
        exit_idx, exit_price, reason = exit_trade(
            times, opens, highs, lows, closes, entry_idx, side, stop, tp, flat_time, slippage
        )

        pnl = (exit_price - entry) * qty * sign
        equity += pnl
        trades_per_day[day] = trades_per_day.get(day, 0) + 1
        busy_until = exit_idx
        trades.append({
            "entrada": times[entry_idx].strftime("%Y-%m-%d %H:%M"),
            "salida": times[exit_idx].strftime("%Y-%m-%d %H:%M"),
            "lado": side,
            "cantidad": qty,
            "precio_entrada": round(entry, 2),
            "stop": stop,
            "take_profit": tp,
            "precio_salida": round(exit_price, 2),
            "motivo": reason,
            "resultado": round(pnl, 2),
            "r": round(pnl / (qty * stop_distance), 2),
            "equity": round(equity, 2),
        })
    return trades, skipped


def metrics(trades, start_equity, frame):
    result = {
        "trades": len(trades),
        "buy_hold_pct": (frame["close"].iloc[-1] / frame["open"].iloc[0] - 1) * 100,
    }
    if not trades:
        return result
    pnls = [t["resultado"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    peak, max_dd, equity = start_equity, 0.0, start_equity
    streak = worst_streak = 0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
        streak = streak + 1 if pnl <= 0 else 0
        worst_streak = max(worst_streak, streak)

    by_reason = {}
    for trade in trades:
        by_reason[trade["motivo"]] = by_reason.get(trade["motivo"], 0) + 1
    by_side = {}
    for side in ("LONG", "SHORT"):
        side_pnls = [t["resultado"] for t in trades if t["lado"] == side]
        if side_pnls:
            by_side[side] = (len(side_pnls), sum(side_pnls), sum(p > 0 for p in side_pnls))
    monthly = {}
    for trade in trades:
        month = trade["entrada"][:7]
        count, total = monthly.get(month, (0, 0.0))
        monthly[month] = (count + 1, total + trade["resultado"])

    total = sum(pnls)
    result.update({
        "total": total,
        "total_pct": total / start_equity * 100,
        "win_rate": len(wins) / len(pnls) * 100,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "profit_factor": sum(wins) / abs(sum(losses)) if sum(losses) else float("inf"),
        "avg_r": sum(t["r"] for t in trades) / len(trades),
        "max_dd": max_dd,
        "worst_streak": worst_streak,
        "best": max(pnls),
        "worst": min(pnls),
        "by_reason": by_reason,
        "by_side": by_side,
        "monthly": monthly,
    })
    return result


def verdict(m):
    if m["trades"] < 30:
        return "⚠️ Pocas operaciones: el resultado no es confiable estadísticamente. Probá con más meses."
    if m["profit_factor"] >= 1.3 and m["total"] > 0:
        return "✅ La estrategia fue rentable en este período."
    if m["total"] > 0:
        return "🟡 Ganancia chica: el margen es bajo y puede desaparecer con costos o mala suerte."
    return "❌ La estrategia perdió plata en este período."


def report_markdown(m, period, args, skipped):
    lines = [
        f"# Backtest NVDA: {period}",
        "",
        f"Datos `{args.feed}`, capital inicial ${args.equity:,.0f}, slippage ${args.slippage:.2f} por acción.",
        "",
    ]
    if not m["trades"]:
        lines.append("No hubo ninguna operación en el período.")
        return "\n".join(lines)
    lines += [
        f"**{verdict(m)}**",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Operaciones | {m['trades']} |",
        f"| Aciertos | {m['win_rate']:.1f}% |",
        f"| Resultado total | {bot.money(m['total'])} ({m['total_pct']:+.2f}%) |",
        f"| Profit factor | {m['profit_factor']:.2f} |",
        f"| R promedio por operación | {m['avg_r']:+.2f} |",
        f"| Ganancia promedio | {bot.money(m['avg_win'])} |",
        f"| Pérdida promedio | {bot.money(m['avg_loss'])} |",
        f"| Mejor / peor operación | {bot.money(m['best'])} / {bot.money(m['worst'])} |",
        f"| Máxima caída (drawdown) | {m['max_dd']:.2f}% |",
        f"| Peor racha de pérdidas | {m['worst_streak']} seguidas |",
        f"| Comprar y mantener NVDA | {m['buy_hold_pct']:+.2f}% |",
        "",
        "## Por lado",
        "",
        "| Lado | Operaciones | Aciertos | Resultado |",
        "|---|---|---|---|",
    ]
    for side, (count, total, wins) in m["by_side"].items():
        lines.append(f"| {side} | {count} | {wins / count * 100:.0f}% | {bot.money(total)} |")
    lines += ["", "## Cómo se cerraron", "", "| Motivo | Operaciones |", "|---|---|"]
    for reason, count in m["by_reason"].items():
        lines.append(f"| {reason} | {count} |")
    lines += ["", "## Por mes", "", "| Mes | Operaciones | Resultado |", "|---|---|---|"]
    for month, (count, total) in sorted(m["monthly"].items()):
        lines.append(f"| {month} | {count} | {bot.money(total)} |")
    lines += [
        "",
        "Señales no tomadas: " + ", ".join(f"{k}: {v}" for k, v in skipped.items()) + ".",
        "",
        "El detalle de cada operación está en el archivo `backtest_trades.csv` (artifact de esta ejecución).",
    ]
    return "\n".join(lines)


def report_telegram(m, period, args):
    if not m["trades"]:
        return f"🧪 Backtest NVDA {period}\nNo hubo ninguna operación en el período."
    return (
        f"🧪 Backtest NVDA {period} (datos {args.feed})\n"
        f"{verdict(m)}\n\n"
        f"Operaciones: {m['trades']}\n"
        f"Aciertos: {m['win_rate']:.1f}%\n"
        f"Resultado: {bot.money(m['total'])} ({m['total_pct']:+.2f}%)\n"
        f"Profit factor: {m['profit_factor']:.2f}\n"
        f"Máxima caída: {m['max_dd']:.2f}%\n"
        f"Peor racha: {m['worst_streak']} pérdidas seguidas\n"
        f"Comprar y mantener NVDA: {m['buy_hold_pct']:+.2f}%\n\n"
        "El informe completo está en GitHub Actions."
    )


def main():
    parser = argparse.ArgumentParser(description="Backtest de la estrategia NVDA del bot.")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--slippage", type=float, default=0.01)
    parser.add_argument("--csv", default="backtest_trades.csv")
    parser.add_argument("--telegram", action="store_true", help="enviar el resumen por Telegram")
    args = parser.parse_args()
    if not 1 <= args.months <= 36:
        parser.error("--months tiene que estar entre 1 y 36")
    if not bot.KEY or not bot.SECRET:
        raise RuntimeError("Faltan ALPACA_KEY o ALPACA_SECRET.")

    # Free Alpaca plans can't query the most recent 15 minutes of SIP data.
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    start = end - pd.DateOffset(months=args.months)
    frame = fetch_bars(start, end, args.feed)
    if len(frame) <= WARMUP_BARS:
        raise RuntimeError(f"Datos insuficientes: {len(frame)} velas.")
    session_closes = fetch_closes(start, end)

    trades, skipped = simulate(frame, session_closes, args.equity, args.slippage)
    m = metrics(trades, args.equity, frame)
    period = f"{frame.index[0]:%Y-%m-%d} a {frame.index[-1]:%Y-%m-%d}"

    with open(args.csv, "w", newline="", encoding="utf-8") as file:
        fields = list(trades[0].keys()) if trades else ["entrada"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trades)

    report = report_markdown(m, period, args, skipped)
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as file:
            file.write(report + "\n")
    if args.telegram:
        bot.telegram(report_telegram(m, period, args))


if __name__ == "__main__":
    main()
