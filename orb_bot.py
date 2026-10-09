"""
Bot de day trading ORB5 en Alpaca PAPER: ruptura del rango de los primeros 5 minutos.

Cada día elige las acciones "en juego" (las de mayor volumen relativo en los primeros 5
minutos), opera la ruptura en la dirección de esa primera vela con el stop-loss en el otro
extremo del rango, y cierra todo 10 minutos antes del cierre. Es la estrategia que mejor
salió en research/ (ver el informe del workflow "research").

Modos (los elige el workflow orb.yml):
  python orb_bot.py session     9:00-11:00 NY: selección y entradas (en media jornada también cierra)
  python orb_bot.py close       15:35-15:55 NY: cierra todo a las 15:50 y manda el resumen
  python orb_bot.py diagnostic  muestra la selección de hoy y la cuenta, sin operar
"""
import csv
import math
import os
import sys
import time

import pandas as pd
import requests

import bot

NY = bot.NY
DATA_URL = bot.DATA_URL

# ------------------------------------------------------------------ configuración
UNIVERSE = [
    "SPY", "QQQ", "NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "MSFT", "GOOGL", "NFLX", "AVGO",
]
TOP_N = 3                 # acciones por día (las de mayor volumen relativo)
RVOL_MIN = 1.0            # volumen de los primeros 5 min vs. su promedio de 14 días
ENTRY_MODE = "cierre1m"   # "cierre1m": entra cuando una vela de 1 min cierra fuera del rango; "market": a las 9:35
STOP_MODE = "or"          # "or": stop en el otro extremo del rango; "atr10": 10% del ATR diario
TP_R = None               # take profit en múltiplos del riesgo (None = sin TP, sale al cierre)
RISK_PCT = 1.0            # % del equity arriesgado por operación
LEV_CAP = 4.0             # exposición total máxima (x equity) repartida entre TOP_N posiciones
ENTRY_DEADLINE_MIN = 11 * 60   # no abre operaciones después de las 11:00 NY
FLATTEN_BEFORE_CLOSE = pd.Timedelta(minutes=10)
MIN_STOP_PCT = 0.0005     # no opera stops más chicos que 0,05% del precio
POLL_SECONDS = 5
ORDER_PREFIX = "orb"
JOURNAL = "orb_journal.csv"


# ------------------------------------------------------------------ datos
def data_get(path, params):
    for attempt in range(6):
        response = bot.SESSION.get(f"{DATA_URL}{path}", headers=bot.HEADERS, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()


def multi_bars(symbols, timeframe, start, end=None, feed="iex"):
    """{symbol: DataFrame[o,h,l,c,v]} indexed by New York bar-start time."""
    params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start, "limit": 10000,
              "adjustment": "split", "feed": feed, "sort": "asc"}
    if end:
        params["end"] = end
    rows = {s: [] for s in symbols}
    while True:
        data = data_get("/v2/stocks/bars", params)
        for sym, bars in (data.get("bars") or {}).items():
            rows.setdefault(sym, []).extend(bars)
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    out = {}
    for sym, bars in rows.items():
        if not bars:
            continue
        df = pd.DataFrame(bars)[["t", "o", "h", "l", "c", "v"]]
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
        out[sym] = df.set_index("t").sort_index()
    return out


def iso(ts):
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def history(symbols, session_open):
    """Per symbol: average first-5-minute volume of the previous 14 sessions and daily ATR(14)."""
    start = session_open - pd.Timedelta(days=30)
    five = multi_bars(symbols, "5Min", iso(start), iso(session_open))
    # Free plans can't query the last 15 minutes of SIP data; yesterday's daily bars are enough.
    sip_end = min(session_open, pd.Timestamp.now(tz=NY)) - pd.Timedelta(minutes=20)
    daily = multi_bars(symbols, "1Day", iso(start - pd.Timedelta(days=10)), iso(sip_end), feed="sip")
    out = {}
    for sym in symbols:
        df = five.get(sym)
        if df is None:
            continue
        first = df[(df.index.hour == 9) & (df.index.minute == 30)]["v"].tail(14)
        d = daily.get(sym)
        atr = None
        if d is not None and len(d) >= 15:
            prev_close = d["c"].shift(1)
            tr = pd.concat([d["h"] - d["l"], (d["h"] - prev_close).abs(), (d["l"] - prev_close).abs()], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())
        if len(first) >= 10:
            out[sym] = {"avg_vol5": float(first.mean()), "atr14": atr}
    return out


def opening_ranges(symbols, session_open, hist):
    """First 5 minutes (1-minute IEX bars) -> candidates ranked by relative volume."""
    last_minute = session_open + pd.Timedelta(minutes=4)
    for _ in range(12):  # the 9:34 bar can take a few seconds to appear
        bars = multi_bars(symbols, "1Min", iso(session_open), iso(session_open + pd.Timedelta(minutes=5)))
        if sum(1 for df in bars.values() if df.index[-1] >= last_minute) >= len(bars) * 0.8:
            break
        time.sleep(5)
    cands = []
    for sym, df in bars.items():
        h = hist.get(sym)
        df = df[df.index < session_open + pd.Timedelta(minutes=5)]
        if h is None or df.empty or h["avg_vol5"] <= 0:
            continue
        o, c = float(df["o"].iloc[0]), float(df["c"].iloc[-1])
        rvol = float(df["v"].sum()) / h["avg_vol5"]
        if c == o or rvol < RVOL_MIN:
            continue
        cands.append({"sym": sym, "side": 1 if c > o else -1, "open": o, "close": c,
                      "high": float(df["h"].max()), "low": float(df["l"].min()), "rvol": rvol, "atr14": h["atr14"]})
    cands.sort(key=lambda x: -x["rvol"])
    return cands[:TOP_N]


# ------------------------------------------------------------------ órdenes
def stop_price(c, fill):
    if STOP_MODE == "or":
        return c["low"] if c["side"] == 1 else c["high"]
    if not c["atr14"]:
        return None
    return fill - c["side"] * 0.10 * c["atr14"]


def enter(c, price, equity, today):
    side = c["side"]
    stop = stop_price(c, price)
    if stop is None:
        return f"{c['sym']}: sin ATR, no se opera"
    stop = round(stop, 2)
    dist = side * (price - stop)
    if dist < MIN_STOP_PCT * price:
        return f"{c['sym']}: stop demasiado cerca ({dist:.2f}), no se opera"
    qty = math.floor(min(equity * RISK_PCT / 100 / dist, equity * LEV_CAP / TOP_N / price))
    if qty < 1:
        return f"{c['sym']}: tamaño menor a 1 acción, no se opera"
    payload = {
        "symbol": c["sym"], "qty": str(qty), "side": "buy" if side == 1 else "sell", "type": "market",
        "time_in_force": "day", "order_class": "bracket" if TP_R else "oto",
        "stop_loss": {"stop_price": f"{stop:.2f}"},
        "client_order_id": f"{ORDER_PREFIX}-{today}-{c['sym']}",
    }
    if TP_R:
        payload["take_profit"] = {"limit_price": f"{price + side * TP_R * dist:.2f}"}
    try:
        bot.api("POST", "/v2/orders", payload=payload)
    except requests.HTTPError as exc:
        return f"{c['sym']}: Alpaca rechazó la orden ({bot.redact(str(exc))[:150]})"
    risk = qty * dist
    return (f"{'🟢 COMPRA' if side == 1 else '🔴 VENTA EN CORTO'} {c['sym']} x{qty} a ~{price:.2f}\n"
            f"   Stop {stop:.2f} (riesgo ${risk:,.0f}){f' | TP {payload['take_profit']['limit_price']}' if TP_R else ''}")


def latest_closed_minutes(symbols, since, now):
    bars = multi_bars(symbols, "1Min", iso(since), iso(now + pd.Timedelta(minutes=1)))
    return {s: df[df.index + pd.Timedelta(minutes=1) <= now] for s, df in bars.items()}


# ------------------------------------------------------------------ sesiones
def wait_until(ts):
    while True:
        left = (ts - pd.Timestamp.now(tz=NY)).total_seconds()
        if left <= 0:
            return
        time.sleep(min(left, 30))


def today_session():
    """(open, close) of today's session from Alpaca's calendar, or None if the market is closed today."""
    today = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    days = bot.api("GET", "/v2/calendar", params={"start": today, "end": today})
    if not days:
        return None
    d = days[0]
    return (pd.Timestamp(f"{d['date']} {d['open']}").tz_localize(NY),
            pd.Timestamp(f"{d['date']} {d['close']}").tz_localize(NY))


def run_session():
    sess = today_session()
    now = pd.Timestamp.now(tz=NY)
    if sess is None:
        print("Hoy no hay mercado.")
        return
    session_open, session_close = sess
    # Several crons cover summer and winter time and GitHub's delays; only a run that starts
    # before the 9:35 selection works, and a queued duplicate starts after it and exits here.
    if not (session_open - pd.Timedelta(minutes=50) <= now <= session_open + pd.Timedelta(minutes=4)):
        print(f"Fuera de la ventana de la sesión ({now:%H:%M} NY); sale sin hacer nada.")
        return
    today = session_open.strftime("%Y%m%d")
    hist = history(UNIVERSE, session_open)
    wait_until(session_open + pd.Timedelta(minutes=5, seconds=3))
    cands = opening_ranges(UNIVERSE, session_open, hist)
    account = bot.api("GET", "/v2/account")
    equity = float(account["equity"])
    if not cands:
        bot.telegram("📭 ORB5: hoy ninguna acción del universo tuvo volumen relativo suficiente. No se opera.")
        return
    lines = [f"🎯 ORB5 en juego hoy ({len(cands)}):"]
    for c in cands:
        lines.append(f"• {c['sym']} {'alcista' if c['side'] == 1 else 'bajista'} | rango {c['low']:.2f}-{c['high']:.2f} "
                     f"| volumen x{c['rvol']:.1f}")
    lines.append("Entra al confirmar la ruptura con una vela de 1 minuto (hasta las 11:00)."
                 if ENTRY_MODE == "cierre1m" else "Entrando a mercado ahora.")
    bot.telegram("\n".join(lines))

    deadline = min(session_open.replace(hour=ENTRY_DEADLINE_MIN // 60, minute=ENTRY_DEADLINE_MIN % 60),
                   session_close - FLATTEN_BEFORE_CLOSE)
    pending = {c["sym"]: c for c in cands}
    if ENTRY_MODE == "market":
        for c in cands:
            bot.telegram("📌 ORB5 " + enter(c, c["close"], equity, today))
        pending = {}
    while pending and pd.Timestamp.now(tz=NY) < deadline:
        now = pd.Timestamp.now(tz=NY)
        bars = latest_closed_minutes(list(pending), session_open + pd.Timedelta(minutes=5), now)
        for sym in list(pending):
            c, df = pending[sym], bars.get(sym)
            if df is None or df.empty:
                continue
            level = c["high"] if c["side"] == 1 else c["low"]
            broke = df[(df["c"] > level) if c["side"] == 1 else (df["c"] < level)]
            if broke.empty:
                continue
            bot.telegram("📌 ORB5 " + enter(c, float(df["c"].iloc[-1]), equity, today))
            del pending[sym]
        time.sleep(POLL_SECONDS)
    if pending:
        bot.telegram("⌛ Sin ruptura confirmada antes del límite: " + ", ".join(pending) + ". No se operan.")
    # Half days close at 13:00, before the afternoon job runs: flatten from here.
    if session_close.hour < 15:
        wait_until(session_close - FLATTEN_BEFORE_CLOSE)
        flatten_and_report(session_open)


def our_orders(session_open):
    orders = bot.api("GET", "/v2/orders", params={"status": "all", "after": iso(session_open - pd.Timedelta(hours=6)),
                                                   "limit": 500, "nested": "true"})
    today = session_open.strftime("%Y%m%d")
    return [o for o in orders or [] if (o.get("client_order_id") or "").startswith(f"{ORDER_PREFIX}-{today}-")]


def flatten_and_report(session_open):
    orders = our_orders(session_open)
    symbols = sorted({o["symbol"] for o in orders})
    for o in bot.api("GET", "/v2/orders", params={"status": "open", "limit": 500}) or []:
        if o["symbol"] in symbols:
            bot.api("DELETE", f"/v2/orders/{o['id']}", allow_404=True)
    time.sleep(2)
    for sym in symbols:
        if bot.api("GET", f"/v2/positions/{sym}", allow_404=True):
            bot.api("DELETE", f"/v2/positions/{sym}", allow_404=True)
    time.sleep(8)
    report(session_open, symbols)


def report(session_open, symbols):
    fills = bot.api("GET", "/v2/account/activities/FILL", params={"date": session_open.strftime("%Y-%m-%d")}) or []
    pnl = {}
    for f in fills:
        if f.get("symbol") in symbols:
            sign = -1 if f["side"] in ("buy",) else 1
            pnl[f["symbol"]] = pnl.get(f["symbol"], 0.0) + sign * float(f["qty"]) * float(f["price"])
    account = bot.api("GET", "/v2/account")
    equity, last_equity = float(account["equity"]), float(account.get("last_equity") or account["equity"])
    lines = [f"📊 Resumen ORB5 {session_open:%Y-%m-%d} (PAPER)"]
    if pnl:
        for sym, value in sorted(pnl.items()):
            lines.append(f"• {sym}: {bot.money(value)}")
        lines.append(f"Total operaciones: {bot.money(sum(pnl.values()))}")
    else:
        lines.append("Hoy no hubo operaciones.")
    lines.append(f"Resultado del día (cuenta): {bot.money(equity - last_equity)} ({(equity / last_equity - 1) * 100:+.2f}%)")
    lines.append(f"Equity: ${equity:,.2f}")
    write_journal(session_open, pnl, equity - last_equity, equity)
    still_open = [p["symbol"] for p in bot.api("GET", "/v2/positions") or [] if p["symbol"] in symbols]
    if still_open:
        lines.append("⚠️ Quedaron posiciones abiertas: " + ", ".join(still_open) + ". Revisalas en Alpaca.")
    bot.telegram("\n".join(lines))


def write_journal(session_open, pnl, day_pnl, equity):
    """One row per traded symbol (or one empty row) per day; the workflow commits the file."""
    new = not os.path.exists(JOURNAL)
    with open(JOURNAL, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(["fecha", "simbolo", "resultado", "resultado_dia_cuenta", "equity"])
        date = session_open.strftime("%Y-%m-%d")
        for sym, value in sorted(pnl.items()) or [("", 0.0)]:
            writer.writerow([date, sym, f"{value:.2f}", f"{day_pnl:.2f}", f"{equity:.2f}"])


def run_close():
    sess = today_session()
    now = pd.Timestamp.now(tz=NY)
    if sess is None:
        print("Hoy no hay mercado.")
        return
    session_open, session_close = sess
    flatten_at = session_close - FLATTEN_BEFORE_CLOSE
    if session_close.hour < 15 or not (flatten_at - pd.Timedelta(minutes=40) <= now <= session_close):
        print(f"Fuera de la ventana de cierre ({now:%H:%M} NY); sale sin hacer nada.")
        return
    if now > flatten_at + pd.Timedelta(seconds=30):
        symbols = {o["symbol"] for o in our_orders(session_open)}
        if not any(p["symbol"] in symbols for p in bot.api("GET", "/v2/positions") or []):
            print("Ya se cerró y se informó hoy; sale sin repetir el resumen.")
            return
    wait_until(flatten_at)
    flatten_and_report(session_open)


def run_diagnostic():
    sess = today_session()
    account = bot.api("GET", "/v2/account")
    lines = ["🧪 Diagnóstico ORB5 (PAPER, no opera)", f"Equity: ${float(account['equity']):,.2f}",
             f"Universo: {len(UNIVERSE)} símbolos, top {TOP_N}, volumen relativo >= {RVOL_MIN}",
             f"Entrada: {ENTRY_MODE}, stop: {STOP_MODE}, TP: {TP_R or 'ninguno'}, riesgo {RISK_PCT}% por operación"]
    now = pd.Timestamp.now(tz=NY)
    if sess and now >= sess[0] + pd.Timedelta(minutes=5):
        cands = opening_ranges(UNIVERSE, sess[0], history(UNIVERSE, sess[0]))
        lines.append("Selección de hoy: " + (", ".join(f"{c['sym']} ({'alcista' if c['side'] == 1 else 'bajista'}, "
                                                        f"x{c['rvol']:.1f})" for c in cands) or "ninguna"))
    else:
        lines.append("Todavía no abrió el mercado hoy: la selección se hace a las 9:35 NY.")
    bot.telegram("\n".join(lines))


def main():
    if not bot.KEY or not bot.SECRET:
        raise RuntimeError("Faltan ALPACA_KEY o ALPACA_SECRET en los secrets de GitHub.")
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ORB_MODE", "diagnostic")
    {"session": run_session, "close": run_close, "diagnostic": run_diagnostic}[mode]()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            bot.telegram(f"⚠️ El bot ORB5 falló\n{bot.redact(f'{type(exc).__name__}: {exc}')[:800]}\n\nRevisá el log en GitHub Actions.")
        except Exception as tg_exc:
            print("No se pudo avisar por Telegram:", bot.redact(str(tg_exc)))
        raise
