"""
Bot de day trading por ruptura del rango de apertura (ORB) en Alpaca PAPER.

Cada día, al terminar el rango de apertura (por defecto los primeros 15 minutos), elige las
acciones "en juego": las de mayor volumen relativo en ese rango dentro del universo. Opera la
primera ruptura confirmada (una vela de 1 minuto que cierra fuera del rango) con el stop-loss
en el otro extremo del rango puesto en Alpaca, y cierra todo 10 minutos antes del cierre. Es la
variante más consistente en research/ con datos SIP e IEX (ver el workflow "research").

Modos (los elige el workflow orb.yml):
  python orb_bot.py session     ~9:00-12:00 NY: selección y entradas (en media jornada también cierra)
  python orb_bot.py close       ~15:30-15:55 NY: cierra todo a las 15:50 y manda el resumen
  python orb_bot.py diagnostic  muestra la selección de la última sesión y la cuenta, sin operar
  python orb_bot.py status      posiciones abiertas, resultado del día y operaciones cerradas, sin operar
  python orb_bot.py listen      escucha comandos de Telegram ("estado", "ayuda") durante LISTEN_MINUTES
"""
import csv
import json
import math
import os
import re
import sys
import time

import pandas as pd
import requests

import bot

NY = bot.NY
DATA_URL = bot.DATA_URL

# ------------------------------------------------------------------ configuración
UNIVERSE = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "ORCL", "CRM", "ADBE", "INTC", "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MRVL",
    "ARM", "SMCI", "PLTR", "SNOW", "CRWD", "PANW", "NOW", "SHOP", "UBER", "ABNB", "COIN", "MSTR",
    "HOOD", "SOFI", "RIVN", "LCID", "NIO", "BABA", "PDD", "JD", "DIS", "NKE", "SBUX", "MCD",
    "WMT", "COST", "TGT", "HD", "LOW", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "V", "MA",
    "PYPL", "AXP", "XOM", "CVX", "OXY", "COP", "SLB", "BA", "CAT", "DE", "GE", "LMT", "RTX", "F",
    "GM", "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "MRNA", "BMY", "GILD", "AMGN", "KO", "PEP",
    "PG", "T", "VZ", "TMUS", "CMCSA", "ROKU", "SNAP", "PINS", "RBLX", "DKNG", "AFRM", "UPST",
    "MARA", "RIOT", "DELL", "ANET",
]
OR_MINUTES = 15           # rango de apertura: primeros 15 minutos
BOTH_SIDES = True         # True: ruptura hacia cualquier lado; False: solo en la dirección de la primera vela
CONFIRM_BARS = 1          # velas de 1 minuto seguidas que tienen que cerrar fuera del rango antes de entrar
                          # (2 se probó en research/: mismos aciertos, algo menos de ganancia promedio)
TOP_N = 3                 # acciones por día (las de mayor volumen relativo)
RVOL_MIN = 0.0            # mínimo de volumen del rango vs. su promedio de 14 sesiones (0 = siempre las TOP_N)
STOP_MODE = "opp"         # "opp": otro extremo del rango; "mid": mitad del rango; "atr10": 10% del ATR diario
TP_R = None               # take profit en múltiplos del riesgo (None = sin TP, sale al cierre)
RISK_PCT = 1.0            # % del equity arriesgado por operación
LEV_CAP = 4.0             # exposición total máxima (x equity), repartida entre TOP_N posiciones
ENTRY_DEADLINE_MIN = 12 * 60   # no abre operaciones después de las 12:00 NY
FLATTEN_BEFORE_CLOSE = pd.Timedelta(minutes=10)
MIN_STOP_PCT = 0.0005     # no opera stops más chicos que 0,05% del precio
POLL_SECONDS = 5
LISTEN_MINUTES = 14       # cada corrida del workflow telegram.yml escucha este tiempo
ORDER_PREFIX = "orb"
JOURNAL = "orb_journal.csv"
NAME = f"ORB{OR_MINUTES}"


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
    rows = {}
    for i in range(0, len(symbols), 100):  # keep the URL short
        params = {"symbols": ",".join(symbols[i:i + 100]), "timeframe": timeframe, "start": start,
                  "limit": 10000, "adjustment": "split", "feed": feed, "sort": "asc"}
        if end:
            params["end"] = end
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
        df = pd.DataFrame(bars)[["t", "o", "h", "l", "c", "v"]]
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
        out[sym] = df.set_index("t").sort_index()
    return out


def iso(ts):
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def history(symbols, session_open):
    """Per symbol: average opening-range volume of the previous 14 sessions (and daily ATR if needed)."""
    start = session_open - pd.Timedelta(days=30)
    five = multi_bars(symbols, "5Min", iso(start), iso(session_open))
    daily = {}
    if STOP_MODE == "atr10":
        # Free plans can't query the last 15 minutes of SIP data; yesterday's daily bars are enough.
        sip_end = min(session_open, pd.Timestamp.now(tz=NY)) - pd.Timedelta(minutes=20)
        daily = multi_bars(symbols, "1Day", iso(start - pd.Timedelta(days=10)), iso(sip_end), feed="sip")
    out = {}
    for sym in symbols:
        df = five.get(sym)
        if df is None:
            continue
        minute = df.index.hour * 60 + df.index.minute
        in_range = df[(minute >= 570) & (minute < 570 + OR_MINUTES)]
        vols = in_range.groupby(in_range.index.date)["v"].sum().tail(14)
        atr = None
        d = daily.get(sym)
        if d is not None and len(d) >= 15:
            prev_close = d["c"].shift(1)
            tr = pd.concat([d["h"] - d["l"], (d["h"] - prev_close).abs(), (d["l"] - prev_close).abs()], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())
        if len(vols) >= 10:
            out[sym] = {"avg_vol": float(vols.mean()), "atr14": atr}
    return out


def opening_ranges(symbols, session_open, hist):
    """Opening range from 1-minute IEX bars -> the TOP_N candidates by relative volume."""
    range_end = session_open + pd.Timedelta(minutes=OR_MINUTES)
    last_minute = range_end - pd.Timedelta(minutes=1)
    for _ in range(12):  # the last bar of the range can take a few seconds to appear
        bars = multi_bars(symbols, "1Min", iso(session_open), iso(range_end))
        if sum(1 for df in bars.values() if df.index[-1] >= last_minute) >= len(bars) * 0.8:
            break
        time.sleep(5)
    cands = []
    for sym, df in bars.items():
        h = hist.get(sym)
        df = df[df.index < range_end]
        if h is None or df.empty or h["avg_vol"] <= 0:
            continue
        o, c = float(df["o"].iloc[0]), float(df["c"].iloc[-1])
        rvol = float(df["v"].sum()) / h["avg_vol"]
        if rvol < RVOL_MIN or (not BOTH_SIDES and c == o):
            continue
        cands.append({"sym": sym, "side": None if BOTH_SIDES else (1 if c > o else -1), "open": o, "close": c,
                      "high": float(df["h"].max()), "low": float(df["l"].min()), "rvol": rvol, "atr14": h["atr14"]})
    cands.sort(key=lambda x: -x["rvol"])
    return cands[:TOP_N]


def describe(c):
    direction = "cualquier lado" if c["side"] is None else ("alcista" if c["side"] == 1 else "bajista")
    return f"• {c['sym']} ({direction}) | rango {c['low']:.2f}-{c['high']:.2f} | volumen x{c['rvol']:.1f}"


# ------------------------------------------------------------------ órdenes
def stop_price(c, side, price):
    if STOP_MODE == "mid":
        return (c["high"] + c["low"]) / 2
    if STOP_MODE in ("opp", "or"):
        return c["low"] if side == 1 else c["high"]
    if not c["atr14"]:
        return None
    return price - side * 0.10 * c["atr14"]


def enter(c, side, price, equity, today):
    stop = stop_price(c, side, price)
    if stop is None:
        return f"{c['sym']}: sin ATR, no se opera"
    stop = round(stop, 2)
    dist = side * (price - stop)
    if dist < MIN_STOP_PCT * price:
        return f"{c['sym']}: el stop quedó demasiado cerca ({dist:.2f}), no se opera"
    qty = math.floor(min(equity * RISK_PCT / 100 / dist, equity * LEV_CAP / TOP_N / price))
    if qty < 1:
        return f"{c['sym']}: tamaño menor a 1 acción, no se opera"
    payload = {
        "symbol": c["sym"], "qty": str(qty), "side": "buy" if side == 1 else "sell", "type": "market",
        "time_in_force": "day", "order_class": "bracket" if TP_R else "oto",
        "stop_loss": {"stop_price": f"{stop:.2f}"},
        "client_order_id": f"{ORDER_PREFIX}-{today}-{c['sym']}",
    }
    tp_text = ""
    if TP_R:
        payload["take_profit"] = {"limit_price": f"{price + side * TP_R * dist:.2f}"}
        tp_text = f" | TP {payload['take_profit']['limit_price']}"
    try:
        bot.api("POST", "/v2/orders", payload=payload)
    except requests.HTTPError as exc:
        detail = exc.response.text[:200] if exc.response is not None else str(exc)
        return f"{c['sym']}: Alpaca rechazó la orden ({bot.redact(detail)})"
    return (f"{'🟢 COMPRA' if side == 1 else '🔴 VENTA EN CORTO'} {c['sym']} x{qty} a ~{price:.2f}\n"
            f"   Stop {stop:.2f} (riesgo ${qty * dist:,.0f}){tp_text}")


def first_run(flags, n):
    """Position where the first run of n consecutive True values ends, or None."""
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= n:
            return i
    return None


def breakout(c, df):
    """CONFIRM_BARS consecutive 1-minute closes outside the range -> (side, latest close, confirming minute)."""
    up = (df["c"] > c["high"]).tolist() if c["side"] != -1 else []
    down = (df["c"] < c["low"]).tolist() if c["side"] != 1 else []
    i_up, i_down = first_run(up, CONFIRM_BARS), first_run(down, CONFIRM_BARS)
    if i_up is None and i_down is None:
        return None
    side, i = (1, i_up) if i_down is None or (i_up is not None and i_up <= i_down) else (-1, i_down)
    return side, float(df["c"].iloc[-1]), df.index[i]


# ------------------------------------------------------------------ sesiones
def wait_until(ts):
    while True:
        left = (ts - pd.Timestamp.now(tz=NY)).total_seconds()
        if left <= 0:
            return
        time.sleep(min(left, 30))


def sessions(start, end):
    days = bot.api("GET", "/v2/calendar", params={"start": start, "end": end}) or []
    return [(pd.Timestamp(f"{d['date']} {d['open']}").tz_localize(NY),
             pd.Timestamp(f"{d['date']} {d['close']}").tz_localize(NY)) for d in days]


def today_session():
    today = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    found = sessions(today, today)
    return found[0] if found else None


def run_session():
    sess = today_session()
    now = pd.Timestamp.now(tz=NY)
    if sess is None:
        print("Hoy no hay mercado.")
        return
    session_open, session_close = sess
    range_end = session_open + pd.Timedelta(minutes=OR_MINUTES)
    deadline = min(session_open.replace(hour=ENTRY_DEADLINE_MIN // 60, minute=ENTRY_DEADLINE_MIN % 60),
                   session_close - FLATTEN_BEFORE_CLOSE)
    # Several crons cover summer and winter time; the one that is not near the open exits here,
    # and so does a queued duplicate that only starts when the first run is done.
    if not (session_open - pd.Timedelta(minutes=50) <= now <= deadline - pd.Timedelta(minutes=5)):
        print(f"Fuera de la ventana de la sesión ({now:%H:%M} NY); sale sin hacer nada.")
        return
    # If GitHub started this run late, only breakouts from now on count: never chase an old one.
    watch_from = max(range_end, now.floor("min"))
    today = session_open.strftime("%Y%m%d")
    hist = history(UNIVERSE, session_open)
    wait_until(range_end + pd.Timedelta(seconds=3))
    cands = opening_ranges(UNIVERSE, session_open, hist)
    equity = float(bot.api("GET", "/v2/account")["equity"])
    if not cands:
        bot.telegram(f"📭 {NAME}: hoy ninguna acción del universo tuvo volumen relativo suficiente. No se opera.")
        return
    already = {o["symbol"] for o in our_orders(session_open)}
    if already:
        print(f"Ya se operaron hoy: {', '.join(sorted(already))}; sigue solo con el resto.")
    else:
        bot.telegram("\n".join([f"🎯 {NAME} en juego hoy ({len(cands)}):"] + [describe(c) for c in cands] + [
            ("Entra cuando una vela de 1 minuto cierre fuera del rango" if CONFIRM_BARS == 1 else
         f"Entra cuando {CONFIRM_BARS} velas de 1 minuto seguidas cierren fuera del rango")
        + f" (hasta las {deadline:%H:%M} NY)."]))

    pending = {c["sym"]: c for c in cands if c["sym"] not in already}
    while pending and pd.Timestamp.now(tz=NY) < deadline:
        now = pd.Timestamp.now(tz=NY)
        bars = multi_bars(list(pending), "1Min", iso(range_end), iso(now + pd.Timedelta(minutes=1)))
        for sym in list(pending):
            df = bars.get(sym)
            if df is None:
                continue
            closed = df[df.index + pd.Timedelta(minutes=1) <= now]
            got = breakout(pending[sym], closed)
            if got:
                side, price, when = got
                if when < watch_from:
                    pending.pop(sym)
                    bot.telegram(f"⏭️ {NAME} {sym}: rompió el rango antes de que arrancara el bot; no se persigue.")
                    continue
                bot.telegram(f"📌 {NAME} " + enter(pending.pop(sym), side, price, equity, today))
        time.sleep(POLL_SECONDS)
    if pending and pd.Timestamp.now(tz=NY) >= deadline:
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
    symbols = sorted({o["symbol"] for o in our_orders(session_open)})
    for o in bot.api("GET", "/v2/orders", params={"status": "open", "limit": 500}) or []:
        if o["symbol"] in symbols:
            bot.api("DELETE", f"/v2/orders/{o['id']}", allow_404=True)
    time.sleep(2)
    for sym in symbols:
        if bot.api("GET", f"/v2/positions/{sym}", allow_404=True):
            bot.api("DELETE", f"/v2/positions/{sym}", allow_404=True)
    # Wait until the closing orders have really filled before adding up the results.
    for _ in range(45):
        if not open_positions(symbols):
            break
        time.sleep(2)
    report(session_open, symbols)


def open_positions(symbols):
    return {p["symbol"]: p for p in bot.api("GET", "/v2/positions") or [] if p["symbol"] in symbols}


def report(session_open, symbols):
    """Result per symbol: realized from today's fills for closed ones; open ones are listed apart
    with their unrealized result, because a half-closed position's cash flow is not a result."""
    still_open = open_positions(symbols)
    fills = bot.api("GET", "/v2/account/activities/FILL", params={"date": session_open.strftime("%Y-%m-%d")}) or []
    pnl = {}
    for f in fills:
        if f.get("symbol") in symbols and f["symbol"] not in still_open:
            sign = -1 if f["side"] == "buy" else 1
            pnl[f["symbol"]] = pnl.get(f["symbol"], 0.0) + sign * float(f["qty"]) * float(f["price"])
    account = bot.api("GET", "/v2/account")
    equity, last_equity = float(account["equity"]), float(account.get("last_equity") or account["equity"])
    lines = [f"📊 Resumen {NAME} {session_open:%Y-%m-%d} (PAPER)"]
    if pnl:
        lines += [f"• {sym}: {bot.money(value)}" for sym, value in sorted(pnl.items())]
        lines.append(f"Total operaciones: {bot.money(sum(pnl.values()))}")
    else:
        lines.append("Hoy no hubo operaciones.")
    lines.append(f"Resultado del día (cuenta): {bot.money(equity - last_equity)} ({(equity / last_equity - 1) * 100:+.2f}%)")
    lines.append(f"Equity: ${equity:,.2f}")
    write_journal(session_open, pnl, equity - last_equity, equity)
    if still_open:
        lines.append("⚠️ Quedaron posiciones abiertas (no se pudieron cerrar): " + ", ".join(
            f"{sym} {bot.money(float(p['unrealized_pl']))}" for sym, p in still_open.items()) + ". Revisalas en Alpaca.")
    bot.telegram("\n".join(lines))


def journal_has(date):
    """Whether the journal (synced from main by the workflow) already has this day's report."""
    if not os.path.exists(JOURNAL):
        return False
    with open(JOURNAL, encoding="utf-8") as file:
        return any(row and row[0] == date for row in csv.reader(file))


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
    if journal_has(session_open.strftime("%Y-%m-%d")):
        print("El resumen de hoy ya está en el registro; sale sin repetirlo.")
        return
    wait_until(flatten_at)
    flatten_and_report(session_open)


def run_diagnostic():
    now = pd.Timestamp.now(tz=NY)
    account = bot.api("GET", "/v2/account")
    lines = [f"🧪 Diagnóstico {NAME} (PAPER, no opera)", f"Equity: ${float(account['equity']):,.2f}",
             f"Universo: {len(UNIVERSE)} símbolos | top {TOP_N} por volumen relativo"
             + (f" (mínimo x{RVOL_MIN})" if RVOL_MIN else ""),
             f"Stop: {STOP_MODE} | TP: {f'{TP_R}R' if TP_R else 'ninguno, sale al cierre'} | riesgo {RISK_PCT}% "
             f"por operación | entradas hasta "
             f"{ENTRY_DEADLINE_MIN // 60}:{ENTRY_DEADLINE_MIN % 60:02d} NY"]
    past = [s for s in sessions((now - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
            if s[0] + pd.Timedelta(minutes=OR_MINUTES) <= now]
    if past:
        session_open = past[-1][0]
        cands = opening_ranges(UNIVERSE, session_open, history(UNIVERSE, session_open))
        lines.append(f"Selección de la sesión del {session_open:%Y-%m-%d}:")
        lines += [describe(c) for c in cands] or ["(ninguna acción con volumen relativo suficiente)"]
    bot.telegram("\n".join(lines))


def run_status():
    """Open positions with their unrealized result, today's closed trades and the day's result."""
    now = pd.Timestamp.now(tz=NY)
    account = bot.api("GET", "/v2/account")
    equity, last_equity = float(account["equity"]), float(account.get("last_equity") or account["equity"])
    positions = bot.api("GET", "/v2/positions") or []
    lines = [f"📈 Estado {NAME} (PAPER) {now:%Y-%m-%d %H:%M} NY",
             f"Equity: ${equity:,.2f} | Día: {bot.money(equity - last_equity)} ({(equity / last_equity - 1) * 100:+.2f}%)"]
    open_symbols = set()
    if positions:
        lines.append("Abiertas:")
        for p in positions:
            qty = float(p["qty"])
            open_symbols.add(p["symbol"])
            lines.append(f"• {p['symbol']} {'largo' if qty > 0 else 'corto'} x{abs(qty):g}: entrada "
                         f"{float(p['avg_entry_price']):.2f}, ahora {float(p['current_price']):.2f} -> "
                         f"{bot.money(float(p['unrealized_pl']))} ({float(p['unrealized_plpc']) * 100:+.2f}%)")
    else:
        lines.append("Sin posiciones abiertas.")
    fills = bot.api("GET", "/v2/account/activities/FILL", params={"date": now.strftime("%Y-%m-%d")}) or []
    closed = {}
    for f in fills:
        if f.get("symbol") not in open_symbols:
            sign = -1 if f["side"] == "buy" else 1
            closed[f["symbol"]] = closed.get(f["symbol"], 0.0) + sign * float(f["qty"]) * float(f["price"])
    if closed:
        lines.append("Cerradas hoy:")
        lines += [f"• {sym}: {bot.money(value)}" for sym, value in sorted(closed.items())]
    print("\n".join(lines))
    bot.telegram("\n".join(lines))


HELP = ("🤖 Comandos:\n"
        "• estado: posiciones abiertas, operaciones cerradas y resultado de hoy\n"
        "• ayuda: esta lista\n"
        "Respondo de lunes a viernes de 7 a 19 hs de Nueva York.")


def telegram_updates(offset, timeout):
    params = {"timeout": timeout, "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        params["offset"] = offset
    response = bot.SESSION.get(f"https://api.telegram.org/bot{bot.TG_TOKEN}/getUpdates",
                               params=params, timeout=timeout + 15)
    response.raise_for_status()
    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getUpdates: {data}")
    return data["result"]


def handle_command(text):
    word = re.sub(r"[^a-záéíóúñ]", "", (text or "").strip().lower().split("@")[0].split(" ")[0])
    if word in ("estado", "status"):
        run_status()
    elif word in ("ayuda", "help", "start", "comandos"):
        bot.telegram(HELP)


def run_listen():
    """Long-poll Telegram for commands from the configured chat; the workflow restarts it every few minutes."""
    end = time.monotonic() + LISTEN_MINUTES * 60
    offset = None
    while time.monotonic() < end:
        try:
            updates = telegram_updates(offset, int(min(50, max(1, end - time.monotonic()))))
        except Exception as exc:  # network blips: wait and keep listening
            print("getUpdates falló:", bot.redact(str(exc))[:300])
            time.sleep(5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message") or {}
            if str((message.get("chat") or {}).get("id")) != str(bot.TG_CHAT):
                continue  # only the owner's chat can run commands
            try:
                handle_command(message.get("text"))
            except Exception as exc:
                bot.telegram(f"⚠️ No pude responder el comando: {bot.redact(f'{type(exc).__name__}: {exc}')[:300]}")
    if offset is not None:
        telegram_updates(offset, 0)  # confirm the last batch so the next run does not repeat it


def main():
    if not bot.KEY or not bot.SECRET:
        raise RuntimeError("Faltan ALPACA_KEY o ALPACA_SECRET en los secrets de GitHub.")
    mode = sys.argv[1] if len(sys.argv) > 1 else "diagnostic"
    {"session": run_session, "close": run_close, "diagnostic": run_diagnostic, "status": run_status,
     "listen": run_listen}[mode]()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            bot.telegram(f"⚠️ El bot {NAME} falló\n{bot.redact(f'{type(exc).__name__}: {exc}')[:800]}\n\n"
                         "Revisá el log en GitHub Actions.")
        except Exception as tg_exc:
            print("No se pudo avisar por Telegram:", bot.redact(str(tg_exc)))
        raise
