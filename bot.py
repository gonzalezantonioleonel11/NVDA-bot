import os, json, math
from datetime import time as dtime
import requests
import pandas as pd

KEY = os.environ["ALPACA_KEY"]
SECRET = os.environ["ALPACA_SECRET"]
TG_TOKEN = os.environ["TELEGRAM_TOKEN"]
TG_CHAT = os.environ["TELEGRAM_CHAT_ID"]
MANUAL = os.environ.get("MANUAL", "false") == "true"

SYMBOL = "NVDA"
RISK_PCT = 0.5
ATR_MULT = 1.5
RR = 2.0
ADX_MIN = 20
MAX_SIGNALS_DAY = 3
MAX_DRIFT = 0.5
STATE_FILE = "estado.json"
NY = "America/New_York"
H = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}


def telegram(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text},
        timeout=20,
    )
    r.raise_for_status()


def get_bars():
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://data.alpaca.markets/v2/stocks/{SYMBOL}/bars"
    params = {"timeframe": "5Min", "start": start, "limit": 10000,
              "adjustment": "raw", "feed": "iex"}
    bars = []
    while True:
        r = requests.get(url, headers=H, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        bars += data.get("bars") or []
        tok = data.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
    df = pd.DataFrame(bars)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df.set_index("t")[["open", "high", "low", "close", "volume"]]


def get_equity():
    try:
        r = requests.get("https://paper-api.alpaca.markets/v2/account", headers=H, timeout=20)
        r.raise_for_status()
        return float(r.json()["equity"])
    except Exception:
        return 10000.0


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema_fast"] = c.ewm(span=20, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=50, adjust=False).mean()
    typical = (h + l + c) / 3
    day = df.index.date
    df["vwap"] = (typical * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    d = c.diff()
    df["rsi"] = 100 - 100 / (1 + rma(d.clip(lower=0), 14) / rma((-d).clip(lower=0), 14))
    df["rsi_prev"] = df["rsi"].shift()
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    uph, dnl = h.diff(), -l.diff()
    plus_dm = uph.where((uph > dnl) & (uph > 0), 0.0)
    minus_dm = dnl.where((dnl > uph) & (dnl > 0), 0.0)
    plus_di = 100 * rma(plus_dm, 14) / df["atr"]
    minus_di = 100 * rma(minus_dm, 14) / df["atr"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = rma(dx, 14)
    df["vol_ok"] = v > v.rolling(20).mean()
    return df


def evaluate(r):
    base = r["adx"] > ADX_MIN and bool(r["vol_ok"])
    long_ = base and r["ema_fast"] > r["ema_slow"] and r["close"] > r["vwap"] \
        and r["rsi_prev"] <= 45 < r["rsi"] and r["hist"] > 0
    short = base and r["ema_fast"] < r["ema_slow"] and r["close"] < r["vwap"] \
        and r["rsi_prev"] >= 55 > r["rsi"] and r["hist"] < 0
    return "LONG" if long_ else ("SHORT" if short else None)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"date": "", "count": 0, "last_bar": ""}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


def main():
    df = get_bars()
    now = pd.Timestamp.now(tz="UTC")
    if df.empty:
        if MANUAL:
            telegram("⚠️ El bot se conectó pero no recibió velas de NVDA.")
        return
    df = df.between_time("09:30", "15:59")
    if df.empty:
        if MANUAL:
            telegram("⚠️ Sin velas del horario de mercado todavía.")
        return
    price_now = float(df["close"].iloc[-1])
    last_open = df.index[-1]
    closed = df[df.index + pd.Timedelta(minutes=5) <= now].copy()

    if MANUAL:
        telegram(f"✅ Bot conectado. Última vela NVDA: {last_open.strftime('%d/%m %H:%M')} (hora NY), "
                 f"precio {price_now:.2f}. Velas cargadas: {len(df)}.")
        return

    if now - last_open > pd.Timedelta(minutes=30) or len(closed) < 60:
        return

    closed = add_indicators(closed)
    state = load_state()
    today = now.tz_convert(NY).strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "count": 0, "last_bar": state.get("last_bar", "")}

    cand = None
    for t, r in closed.iloc[-3:].iterrows():
        if not (dtime(9, 45) <= t.time() < dtime(15, 45)):
            continue
        side = evaluate(r)
        if side and t.tz_convert("UTC").isoformat() > state["last_bar"]:
            cand = (t, r, side)

    if cand is None or state["count"] >= MAX_SIGNALS_DAY:
        save_state(state)
        return

    t, r, side = cand
    price = float(r["close"])
    sd = float(r["atr"]) * ATR_MULT
    equity = get_equity()
    qty = min(math.floor(equity * RISK_PCT / 100 / sd), math.floor(equity / price))
    state["last_bar"] = t.tz_convert("UTC").isoformat()
    hora = (t + pd.Timedelta(minutes=5)).strftime("%H:%M")

    if side == "LONG":
        stop, tp, titulo = price - sd, price + sd * RR, "COMPRAR (LONG)"
    else:
        stop, tp, titulo = price + sd, price - sd * RR, "VENDER EN CORTO (SHORT)"

    if qty < 1:
        save_state(state)
        return

    if abs(price_now - price) > MAX_DRIFT * sd:
        telegram(f"⏰ SEÑAL TARDE - NO ENTRAR\n{titulo} NVDA, vela de las {hora} (NY).\n"
                 f"Precio de la señal: {price:.2f} | Precio ahora: {price_now:.2f}\n"
                 f"El precio ya se alejó demasiado.")
    else:
        state["count"] += 1
        telegram(f"{titulo} NVDA\nSeñal de la vela de las {hora} (hora NY)\n"
                 f"Cantidad: {qty}\nPrecio señal: {price:.2f} (ahora {price_now:.2f})\n"
                 f"STOP LOSS: {stop:.2f}\nTAKE PROFIT: {tp:.2f}\n"
                 f"Si ya tenés una posición abierta, ignorala. Cerrar todo antes de las 15:45 NY.")
    save_state(state)


main()
