import os
import re
import json
import math
import time
import secrets
from datetime import time as dtime

import pandas as pd
import requests

import levels as sr

# SAFETY: this version is hard-wired to Alpaca PAPER trading.
TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
SYMBOL = os.environ.get("SYMBOL", "NVDA").upper()
NY = "America/New_York"
STATE_FILE = "estado.json"

# Which directions the bot may propose: BOTH, LONG or SHORT.
TRADE_SIDES = os.environ.get("TRADE_SIDES", "BOTH").strip().upper() or "BOTH"
if TRADE_SIDES not in ("BOTH", "LONG", "SHORT"):
    print(f"TRADE_SIDES={TRADE_SIDES!r} no es válido; se usa BOTH.")
    TRADE_SIDES = "BOTH"

# Risk controls
RISK_PCT = 0.5                 # risk budget as % of account equity per trade
ATR_MULT = 1.5                 # base stop distance = ATR * this multiplier
MAX_STOP_ATR = 2.5             # a stop moved behind a level is never wider than this
RR = 2.0                       # take-profit distance = stop distance * this ratio
MIN_RR = 1.5                   # skip the trade if the next level leaves less room than this
MIN_RR_AFTER_APPROVAL = 1.2    # re-check with the live price after the human delay
ADX_MIN = 20
MAX_TRADES_DAY = 3
MAX_NOTIONAL_PCT = 20.0        # cap exposure to 20% of equity per position
MAX_DRIFT_ATR = 0.5            # reject if price moved too far from signal close
ENTRY_START = dtime(9, 45)
ENTRY_END = dtime(15, 20)
FORCE_FLAT_TIME = dtime(15, 50)
APPROVAL_TIMEOUT = 180         # seconds to wait for the Telegram button

# Confluence scoring: trend filters are mandatory, then at least MIN_SCORE of the
# 11 confirmations must agree, and at least one of them must be a trigger.
MIN_SCORE = 6
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Present-time market context
MARKET_SYMBOLS = ("SPY", "QQQ", "SMH")   # S&P 500, Nasdaq 100, semiconductors
NEWS_HOURS = 24
NEWS_HOT_MINUTES = 30

KEY = os.environ.get("ALPACA_KEY", "")
SECRET = os.environ.get("ALPACA_SECRET", "")
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
MANUAL = os.environ.get("MANUAL", "false").lower() == "true"

HEADERS = {
    "APCA-API-KEY-ID": KEY,
    "APCA-API-SECRET-KEY": SECRET,
    "Content-Type": "application/json",
}
SESSION = requests.Session()


def telegram(message):
    if not TG_TOKEN or not TG_CHAT:
        raise RuntimeError("Falta TELEGRAM_TOKEN o TELEGRAM_CHAT_ID")

    response = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": message},
        timeout=20,
    )

    print("Telegram HTTP:", response.status_code)
    print("Telegram respuesta:", response.text[:500])
    response.raise_for_status()


def api(method, path, *, params=None, payload=None, allow_404=False):
    response = SESSION.request(
        method,
        f"{TRADING_URL}{path}",
        headers=HEADERS,
        params=params,
        json=payload,
        timeout=25,
    )
    if allow_404 and response.status_code == 404:
        return None
    if not response.ok:
        print(f"Alpaca {method} {path} -> {response.status_code}: {response.text[:500]}")
    response.raise_for_status()
    if not response.text:
        return {}
    return response.json()


def get_clock():
    return api("GET", "/v2/clock")


def get_account():
    # Fail closed: never substitute a guessed account balance.
    account = api("GET", "/v2/account")
    if account.get("trading_blocked"):
        raise RuntimeError("Alpaca reports trading_blocked=true.")
    equity = float(account["equity"])
    buying_power = float(account.get("buying_power", 0))
    if equity <= 0:
        raise RuntimeError("Account equity is not positive.")
    return equity, buying_power


def _bars_frame(bars):
    frame = pd.DataFrame(bars)
    if frame.empty:
        return frame
    frame["t"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(NY)
    frame = frame.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame = frame.set_index("t")[["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def get_bars_multi(symbols, days=10):
    start = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "5Min",
        "start": start,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
    }
    collected = {symbol: [] for symbol in symbols}
    while True:
        response = SESSION.get(
            f"{DATA_URL}/v2/stocks/bars", headers=HEADERS, params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        for symbol, bars in (data.get("bars") or {}).items():
            collected.setdefault(symbol, []).extend(bars or [])
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    return {symbol: _bars_frame(bars) for symbol, bars in collected.items()}


def get_bars():
    return get_bars_multi([SYMBOL])[SYMBOL]


def get_latest_price():
    response = SESSION.get(
        f"{DATA_URL}/v2/stocks/{SYMBOL}/trades/latest",
        headers=HEADERS,
        params={"feed": "iex"},
        timeout=15,
    )
    response.raise_for_status()
    price = float((response.json().get("trade") or {}).get("p", 0))
    if price <= 0:
        raise RuntimeError("No se pudo obtener el último precio.")
    return price


def market_trend(frame, now_utc):
    """+1 / -1 when the 5-min EMAs and the session VWAP agree, 0 if mixed or no data."""
    if frame.empty:
        return 0
    frame = frame.between_time("09:30", "15:59")
    frame = frame[frame.index + pd.Timedelta(minutes=5) <= now_utc]
    if len(frame) < 60:
        return 0
    close = frame["close"]
    ema_fast = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=50, adjust=False).mean().iloc[-1]
    session = frame[frame.index.date == frame.index[-1].date()]
    volume = session["volume"].sum()
    if volume <= 0:
        return 0
    typical = (session["high"] + session["low"] + session["close"]) / 3
    vwap = (typical * session["volume"]).sum() / volume
    last = close.iloc[-1]
    if ema_fast > ema_slow and last > vwap:
        return 1
    if ema_fast < ema_slow and last < vwap:
        return -1
    return 0


def get_market_context(now_utc):
    """Trend of SPY/QQQ/SMH right now. Informational: failures return None."""
    try:
        frames = get_bars_multi(list(MARKET_SYMBOLS), days=5)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Contexto de mercado no disponible: {exc}")
        return None
    return {
        symbol: market_trend(frames.get(symbol, pd.DataFrame()), now_utc)
        for symbol in MARKET_SYMBOLS
    }


def get_news(now_utc):
    """Latest headlines for SYMBOL. Informational: failures return []."""
    start = (now_utc - pd.Timedelta(hours=NEWS_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        response = SESSION.get(
            f"{DATA_URL}/v1beta1/news",
            headers=HEADERS,
            params={"symbols": SYMBOL, "start": start, "limit": 10, "sort": "desc"},
            timeout=15,
        )
        response.raise_for_status()
        items = response.json().get("news") or []
    except (requests.RequestException, ValueError) as exc:
        print(f"Noticias no disponibles: {exc}")
        return []
    news = []
    for item in items:
        if not item.get("created_at") or not item.get("headline"):
            continue
        news.append({
            "time": pd.to_datetime(item["created_at"], utc=True),
            "headline": " ".join(item["headline"].split()),
            "source": item.get("source", ""),
        })
    return news


def can_short():
    account = api("GET", "/v2/account")
    if not account.get("shorting_enabled", False):
        return False, "La cuenta no tiene habilitada la venta en corto (requiere margen, equity ≥ $2.000)."
    asset = api("GET", f"/v2/assets/{SYMBOL}")
    if not (asset.get("shortable") and asset.get("easy_to_borrow")):
        return False, f"{SYMBOL} no está disponible para venta en corto en este momento."
    return True, ""


def rma(series, period):
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_higher_timeframe_trend(df):
    """1-hour trend built only from completed hours, aligned back onto 5-min bars."""
    hourly = df["close"].resample("1h", label="right", closed="left").last().dropna()
    ema = hourly.ewm(span=20, adjust=False).mean()
    trend = pd.Series(0, index=hourly.index)
    trend[(hourly > ema) & (ema > ema.shift(1))] = 1
    trend[(hourly < ema) & (ema < ema.shift(1))] = -1
    # An hour labelled 11:00 covers 10:00-10:59 and is usable from the 10:55 bar's close
    # (labelled 10:55 in the 5-min frame), i.e. once bar_time + 5min >= hour label.
    bar_close = df.index + pd.Timedelta(minutes=5)
    aligned = trend.reindex(bar_close, method="ffill")
    return pd.Series(aligned.to_numpy(), index=df.index).fillna(0).astype(int)


def add_indicators(frame):
    df = frame.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["ema_fast"] = close.ewm(span=20, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=50, adjust=False).mean()

    typical = (high + low + close) / 3
    session_day = df.index.date
    df["vwap"] = (
        (typical * volume).groupby(session_day).cumsum()
        / volume.groupby(session_day).cumsum().replace(0, float("nan"))
    )

    delta = close.diff()
    avg_gain = rma(delta.clip(lower=0), 14)
    avg_loss = rma((-delta).clip(lower=0), 14)
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_prev"] = df["rsi"].shift(1)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = rma(true_range, 14)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * rma(plus_dm, 14) / df["atr"].replace(0, float("nan"))
    minus_di = 100 * rma(minus_dm, 14) / df["atr"].replace(0, float("nan"))
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = rma(dx, 14)
    df["vol_ok"] = volume > volume.rolling(20, min_periods=20).mean()

    # Confirmed swing points: a pivot is only made available two bars after it forms.
    pivot_high = high.where(high.eq(high.rolling(5, center=True, min_periods=5).max())).shift(2)
    pivot_low = low.where(low.eq(low.rolling(5, center=True, min_periods=5).min())).shift(2)
    df["swing_high"] = pivot_high.ffill()
    df["swing_low"] = pivot_low.ffill()
    prior_swing_high = df["swing_high"].shift(1)
    prior_swing_low = df["swing_low"].shift(1)

    df["sweep_bull"] = (low < prior_swing_low) & (close > prior_swing_low)
    df["sweep_bear"] = (high > prior_swing_high) & (close < prior_swing_high)

    # Three-candle Fair Value Gaps (FVG).
    df["fvg_bull"] = low > high.shift(2)
    df["fvg_bear"] = high < low.shift(2)

    # BOS / CHoCH from breaks of the latest confirmed swing levels.
    break_up = (close > prior_swing_high) & (close.shift(1) <= prior_swing_high)
    break_down = (close < prior_swing_low) & (close.shift(1) >= prior_swing_low)

    event = []
    bias_values = []
    bias = 0
    for up, down in zip(break_up.fillna(False), break_down.fillna(False)):
        if up:
            event.append("CHoCH_UP" if bias == -1 else "BOS_UP")
            bias = 1
        elif down:
            event.append("CHoCH_DOWN" if bias == 1 else "BOS_DOWN")
            bias = -1
        else:
            event.append("")
        bias_values.append(bias)

    df["structure_event"] = event
    df["market_bias"] = bias_values
    df["structure_up_recent"] = df["structure_event"].isin(["BOS_UP", "CHoCH_UP"]).rolling(3).max().fillna(0).astype(bool)
    df["structure_down_recent"] = df["structure_event"].isin(["BOS_DOWN", "CHoCH_DOWN"]).rolling(3).max().fillna(0).astype(bool)
    df["sweep_bull_recent"] = df["sweep_bull"].astype(float).rolling(6, min_periods=1).max().fillna(0).astype(bool)
    df["sweep_bear_recent"] = df["sweep_bear"].astype(float).rolling(6, min_periods=1).max().fillna(0).astype(bool)
    df["fvg_bull_recent"] = df["fvg_bull"].astype(float).rolling(6, min_periods=1).max().fillna(0).astype(bool)
    df["fvg_bear_recent"] = df["fvg_bear"].astype(float).rolling(6, min_periods=1).max().fillna(0).astype(bool)

    df["htf_trend"] = add_higher_timeframe_trend(df)
    df = df.join(sr.fibonacci_columns(df))

    return df


def side_checks(row, side, ctx=None):
    """Checklist for one side: (mandatory, optional, has_trigger)."""
    ctx = ctx or {}
    up = side == "LONG"
    sign = 1 if up else -1
    market = ctx.get("market")

    mandatory = [
        ("ADX > %d (hay tendencia)" % ADX_MIN, row["adx"] > ADX_MIN),
        ("EMA20 %s EMA50" % (">" if up else "<"),
         row["ema_fast"] > row["ema_slow"] if up else row["ema_fast"] < row["ema_slow"]),
        ("Precio %s VWAP" % ("sobre" if up else "bajo"),
         row["close"] > row["vwap"] if up else row["close"] < row["vwap"]),
        ("Tendencia 1H %s" % ("alcista" if up else "bajista"),
         row["htf_trend"] == sign),
    ]
    if market:
        against = sum(1 for value in market.values() if value == -sign)
        mandatory.append(("Mercado general no está todo en contra", against < len(market)))

    rsi_trigger = (
        row["rsi_prev"] <= 45 < row["rsi"] if up else row["rsi_prev"] >= 55 > row["rsi"]
    )
    structure = bool(row["structure_up_recent"] if up else row["structure_down_recent"])
    sweep = bool(row["sweep_bull_recent"] if up else row["sweep_bear_recent"])
    fib_zone = bool(row["fib_bull_zone"] if up else row["fib_bear_zone"])
    reaction = (ctx.get("reaction") or {}).get(side, "")
    if market:
        favor = [symbol for symbol, value in market.items() if value == sign]
        market_check = (
            "Mercado a favor (%d/%d%s)" % (
                len(favor), len(market), ": " + ", ".join(favor) if favor else ""
            ),
            len(favor) >= 2,
        )
    else:
        market_check = ("Mercado general (sin datos)", False)

    optional = [
        ("RSI gatillo (cruce %s)" % ("45↑" if up else "55↓"), rsi_trigger),
        ("Estructura BOS/CHoCH reciente", structure),
        ("Barrido de liquidez reciente", sweep),
        ("Fibonacci: %s en zona dorada 0.5–0.786" % ("rebote" if up else "rechazo"), fib_zone),
        (reaction or ("Reacción en soporte" if up else "Reacción en resistencia"), bool(reaction)),
        ("FVG reciente", bool(row["fvg_bull_recent"] if up else row["fvg_bear_recent"])),
        ("MACD histograma %s" % ("> 0" if up else "< 0"),
         row["macd_hist"] > 0 if up else row["macd_hist"] < 0),
        ("Sesgo de estructura %s" % ("alcista" if up else "bajista"),
         row["market_bias"] == sign),
        ("Volumen sobre promedio", bool(row["vol_ok"])),
        ("RSI sin %s" % ("sobrecompra" if up else "sobreventa"),
         row["rsi"] < RSI_OVERBOUGHT if up else row["rsi"] > RSI_OVERSOLD),
        market_check,
    ]
    mandatory = [(label, bool(ok)) for label, ok in mandatory]
    optional = [(label, bool(ok)) for label, ok in optional]
    has_trigger = bool(rsi_trigger or structure or sweep or fib_zone or reaction)
    return mandatory, optional, has_trigger


def data_ready(row):
    return all(
        pd.notna(row[c]) for c in ("adx", "atr", "vwap", "rsi", "rsi_prev", "ema_fast", "ema_slow")
    ) and row["atr"] > 0


def allowed_sides():
    return ("LONG", "SHORT") if TRADE_SIDES == "BOTH" else (TRADE_SIDES,)


def analyze(row, ctx=None):
    """
    Returns (side, score, (mandatory, optional)). side is None when no setup qualifies.
    Trend filters are mandatory; then at least MIN_SCORE confirmations must agree and
    at least one must be a trigger (RSI cross, BOS/CHoCH, sweep, Fibonacci or S/R).
    """
    if not data_ready(row):
        return None, 0, ([], [])
    best = (None, 0, ([], []))
    for side in allowed_sides():
        mandatory, optional, has_trigger = side_checks(row, side, ctx)
        score = sum(ok for _, ok in optional)
        if all(ok for _, ok in mandatory) and has_trigger and score >= MIN_SCORE and score > best[1]:
            best = (side, score, (mandatory, optional))
    return best


def evaluate(row, ctx=None):
    return analyze(row, ctx)[0]


def build_context(df, market=None):
    """Levels and level reactions known at the close of df's last bar."""
    levels = sr.key_levels(df)
    return {
        "levels": levels,
        "reaction": {side: sr.level_reaction(df, levels, side) for side in ("LONG", "SHORT")},
        "market": market,
    }


def scan(enriched, market=None, last_bar="", lookback=3):
    """Signals on the newest `lookback` completed bars that were not handled before."""
    found = []
    for pos in range(max(0, len(enriched) - lookback), len(enriched)):
        bar_time = enriched.index[pos]
        if not (ENTRY_START <= bar_time.time() < ENTRY_END):
            continue
        bar_iso = bar_time.tz_convert("UTC").isoformat()
        if bar_iso <= last_bar:
            continue
        row = enriched.iloc[pos]
        if not data_ready(row):
            continue
        ctx = build_context(enriched.iloc[:pos + 1], market)
        side, score, checks = analyze(row, ctx)
        if side:
            found.append({
                "time": bar_time, "bar_iso": bar_iso, "row": row, "side": side,
                "score": score, "checks": checks, "ctx": ctx,
            })
    return found


def make_plan(side, entry, atr, levels):
    return sr.plan_trade(
        side, entry, atr, levels,
        atr_mult=ATR_MULT, rr=RR, min_rr=MIN_RR, max_stop_atr=MAX_STOP_ATR,
    )


def position_size(equity, buying_power, price, stop_distance):
    risk_dollars = equity * (RISK_PCT / 100.0)
    max_notional = equity * (MAX_NOTIONAL_PCT / 100.0)
    qty_by_risk = math.floor(risk_dollars / stop_distance)
    qty_by_notional = math.floor(max_notional / price)
    qty_by_buying_power = math.floor(buying_power / price)
    return max(0, min(qty_by_risk, qty_by_notional, qty_by_buying_power)), risk_dollars


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"date": "", "count": 0, "last_bar": ""}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def telegram_api(method, payload):
    if not TG_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN no está configurado; no se puede pedir aprobación.")
    response = SESSION.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error en {method}: {result}")
    return result.get("result")


def ask_approval(signal_text, timeout_seconds=APPROVAL_TIMEOUT):
    """
    Sends inline Approve/Reject buttons and waits for a callback.
    Returns 'approve', 'reject', or 'timeout'.
    Any API issue fails closed: no trade is submitted.
    """
    if not TG_TOKEN or not TG_CHAT:
        raise RuntimeError("Telegram no está configurado; se cancela la operación por seguridad.")

    # Drain old callback updates so a stale button click cannot approve a new trade.
    offset = None
    while True:
        params = {"timeout": 0, "allowed_updates": json.dumps(["callback_query"])}
        if offset is not None:
            params["offset"] = offset
        result = SESSION.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params=params,
            timeout=10,
        )
        result.raise_for_status()
        payload = result.json()
        if not payload.get("ok"):
            raise RuntimeError(f"No se pudieron consultar callbacks de Telegram: {payload}")
        updates = payload.get("result", [])
        if not updates:
            break
        offset = max(update["update_id"] for update in updates) + 1

    token = secrets.token_hex(6)
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ APROBAR OPERACIÓN", "callback_data": f"approve:{token}"},
            {"text": "❌ RECHAZAR", "callback_data": f"reject:{token}"},
        ]]
    }
    minutes = timeout_seconds // 60
    message = telegram_api("sendMessage", {
        "chat_id": TG_CHAT,
        "text": signal_text + f"\n\n⏳ Esperando tu decisión. Si no respondés en {minutes} minutos, no se opera.",
        "reply_markup": keyboard,
    })
    message_id = message["message_id"]
    deadline = time.monotonic() + timeout_seconds
    approver_id = os.environ.get("TELEGRAM_APPROVER_ID", "").strip()

    while time.monotonic() < deadline:
        params = {
            "timeout": 0,
            "allowed_updates": json.dumps(["callback_query"]),
        }
        if offset is not None:
            params["offset"] = offset

        response = SESSION.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            raise RuntimeError(f"Error consultando aprobación de Telegram: {result}")

        updates = result.get("result", [])
        for update in updates:
            offset = max(offset or 0, update["update_id"] + 1)
            callback = update.get("callback_query")
            if not callback:
                continue

            data = callback.get("data", "")
            callback_message = callback.get("message") or {}
            callback_chat = str((callback_message.get("chat") or {}).get("id", ""))
            callback_message_id = callback_message.get("message_id")
            callback_user_id = str((callback.get("from") or {}).get("id", ""))

            # Only accept this exact message/token in the configured chat.
            if data not in (f"approve:{token}", f"reject:{token}"):
                continue
            if callback_chat != str(TG_CHAT) or callback_message_id != message_id:
                telegram_api("answerCallbackQuery", {
                    "callback_query_id": callback["id"],
                    "text": "Esta aprobación no corresponde a esta señal.",
                    "show_alert": True,
                })
                continue
            if approver_id and callback_user_id != approver_id:
                telegram_api("answerCallbackQuery", {
                    "callback_query_id": callback["id"],
                    "text": "Tu usuario no está autorizado para aprobar operaciones.",
                    "show_alert": True,
                })
                continue

            decision = "approve" if data.startswith("approve:") else "reject"
            telegram_api("answerCallbackQuery", {
                "callback_query_id": callback["id"],
                "text": "Aprobada" if decision == "approve" else "Rechazada",
            })
            final_text = signal_text + (
                "\n\n✅ APROBADA. Enviando orden a Alpaca PAPER..."
                if decision == "approve"
                else "\n\n❌ RECHAZADA. No se envió ninguna orden."
            )
            telegram_api("editMessageText", {
                "chat_id": TG_CHAT,
                "message_id": message_id,
                "text": final_text,
                "reply_markup": {"inline_keyboard": []},
            })
            return decision

        time.sleep(2)

    telegram_api("editMessageText", {
        "chat_id": TG_CHAT,
        "message_id": message_id,
        "text": signal_text + f"\n\n⌛ Señal vencida: no se recibió aprobación en {minutes} minutos. No se operó.",
        "reply_markup": {"inline_keyboard": []},
    })
    return "timeout"


def send_order(side, qty, stop, take_profit, client_order_id):
    payload = {
        "symbol": SYMBOL,
        "qty": str(qty),
        "side": "buy" if side == "LONG" else "sell",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{take_profit:.2f}"},
        "stop_loss": {"stop_price": f"{stop:.2f}"},
        "client_order_id": client_order_id,
    }
    # The endpoint is hard-coded to paper-api.alpaca.markets above.
    return api("POST", "/v2/orders", payload=payload)


def wait_for_fill(order_id, timeout_seconds=20):
    deadline = time.monotonic() + timeout_seconds
    order = {}
    while time.monotonic() < deadline:
        order = api("GET", f"/v2/orders/{order_id}")
        if order.get("status") in ("filled", "canceled", "rejected", "expired"):
            break
        time.sleep(2)
    return order


def has_position_or_open_order():
    positions = api("GET", "/v2/positions")
    if any(p.get("symbol") == SYMBOL for p in positions):
        return True, f"Ya existe una posición abierta en {SYMBOL}."
    orders = api("GET", "/v2/orders", params={"status": "open", "symbols": SYMBOL, "limit": 100})
    if orders:
        return True, f"Ya hay una orden abierta para {SYMBOL}."
    return False, ""


def close_position_end_of_day():
    orders = api("GET", "/v2/orders", params={"status": "open", "symbols": SYMBOL, "limit": 100})
    for order in orders or []:
        api("DELETE", f"/v2/orders/{order['id']}")
    position = api("GET", f"/v2/positions/{SYMBOL}", allow_404=True)
    if position:
        api("DELETE", f"/v2/positions/{SYMBOL}")
        telegram(f"🧹 Cierre de fin de día solicitado para {SYMBOL} en PAPER. Verificá en Alpaca que la posición se haya cerrado.")
    else:
        print(f"Fin de día: no hay posición {SYMBOL} abierta.")


SEPARATOR = "━━━━━━━━━━━━━━━━━━"


def format_checks(mandatory, optional):
    lines = ["Filtros obligatorios:"]
    lines += [f"{'✅' if ok else '❌'} {label}" for label, ok in mandatory]
    lines.append("Confirmaciones:")
    lines += [f"{'✅' if ok else '▫️'} {label}" for label, ok in optional]
    return "\n".join(lines)


def format_news(news, now_utc):
    if not news:
        return [f"📰 Sin noticias de {SYMBOL} en las últimas {NEWS_HOURS} h."]
    lines = [f"📰 Noticias ({NEWS_HOURS} h):"]
    for item in news[:3]:
        headline = item["headline"]
        if len(headline) > 110:
            headline = headline[:107] + "..."
        lines.append(f"• {item['time'].tz_convert(NY):%d/%m %H:%M} {headline}")
    if any(now_utc - item["time"] <= pd.Timedelta(minutes=NEWS_HOT_MINUTES) for item in news):
        lines.append(f"⚠️ Hay noticias de los últimos {NEWS_HOT_MINUTES} min: puede haber volatilidad.")
    if any(re.search(r"earnings|quarterly results|guidance", item["headline"], re.I) for item in news):
        lines.append("⚠️ Se habla de resultados/earnings: cuidado con movimientos bruscos.")
    return lines


def format_market_view(ctx, row, price, news, now_utc):
    levels = ctx["levels"]
    above = sr.nearest(levels, price, "up")
    below = sr.nearest(levels, price, "down")
    lines = [
        "🧱 Resistencias: " + (" · ".join(f"${l.price:.2f} {l.label}" for l in above) or "ninguna cercana"),
        "🧱 Soportes: " + (" · ".join(f"${l.price:.2f} {l.label}" for l in below) or "ninguno cercano"),
    ]
    fibs = sr.fib_levels(row)
    if fibs:
        up = row["fib_dir"] == 1
        start, end = (row["fib_lo"], row["fib_hi"]) if up else (row["fib_hi"], row["fib_lo"])
        lines.append(
            f"📐 Fibonacci tramo {'alcista' if up else 'bajista'} ${start:.2f} → ${end:.2f}: "
            + " · ".join(f"{l.label[4:]} ${l.price:.2f}" for l in fibs)
        )
    market = ctx.get("market")
    if market:
        arrows = {1: "▲", -1: "▼", 0: "▬"}
        lines.append("🌎 Mercado ahora: " + " · ".join(f"{s} {arrows[v]}" for s, v in market.items()))
    else:
        lines.append("🌎 Mercado ahora: sin datos")
    lines += format_news(news, now_utc)
    return lines


def format_signal(candidate, plan, qty, equity, news, now_utc):
    side = candidate["side"]
    row = candidate["row"]
    mandatory, optional = candidate["checks"]
    entry, stop, target = plan["entry"], plan["stop"], plan["target"]
    risk, reward = plan["risk"], abs(target - entry)
    if side == "LONG":
        header = f"🟢 LONG — COMPRA {SYMBOL}"
    else:
        header = f"🔴 SHORT — VENTA EN CORTO {SYMBOL}"
    lines = [
        header,
        SEPARATOR,
        f"💵 Entrada: ${entry:.2f} (a mercado)",
        f"🛑 Stop loss: ${stop:.2f} (riesgo ${risk:.2f}/acción · {plan['stop_note']})",
        f"🎯 Take profit: ${target:.2f} (ganancia ${reward:.2f}/acción · {plan['target_note']})",
        f"⚖️ Riesgo/beneficio: 1:{plan['rr']:.1f}",
        f"📦 Cantidad: {qty} acciones (~${qty * entry:,.2f})",
        f"💸 Pérdida máx. si toca el stop: ${qty * risk:,.2f} ({100 * qty * risk / equity:.2f}% de la cuenta)",
        SEPARATOR,
        f"📊 Confluencia {candidate['score']}/{len(optional)} (mínimo {MIN_SCORE})",
        format_checks(mandatory, optional),
        f"RSI {row['rsi']:.1f} · ADX {row['adx']:.1f} · ATR {row['atr']:.2f}",
        SEPARATOR,
        *format_market_view(candidate["ctx"], row, entry, news, now_utc),
        SEPARATOR,
        f"⏰ Vela 5m {candidate['time']:%Y-%m-%d %H:%M} NY · Alpaca PAPER",
    ]
    return "\n".join(lines)


def run_diagnostics(now_utc):
    now_ny = now_utc.tz_convert(NY)
    account_equity, _ = get_account()
    clock = get_clock()
    lines = [
        "🧪 Diagnóstico PAPER solamente",
        f"Cuenta equity: ${account_equity:.2f}",
        f"Mercado abierto según Alpaca: {clock.get('is_open')}",
        f"Hora Nueva York: {now_ny:%Y-%m-%d %H:%M:%S}",
        f"Lados habilitados: {TRADE_SIDES}",
    ]
    frame = get_bars()
    if not frame.empty:
        frame = frame.between_time("09:30", "15:59")
        frame = frame[frame.index + pd.Timedelta(minutes=5) <= now_utc]
    if len(frame) >= 100:
        enriched = add_indicators(frame)
        row = enriched.iloc[-1]
        market = get_market_context(now_utc)
        news = get_news(now_utc)
        ctx = build_context(enriched, market)
        lines.append(f"\nÚltima vela {SYMBOL}: {enriched.index[-1]:%Y-%m-%d %H:%M} cierre ${row['close']:.2f}")
        side, score, (mandatory, optional) = analyze(row, ctx)
        if side:
            plan = make_plan(side, float(row["close"]), float(row["atr"]), ctx["levels"])
            lines.append(f"Señal actual: {side} ({score}/{len(optional)})")
            lines.append(
                f"Stop ${plan['stop']:.2f} · Target ${plan['target']:.2f} · R:R 1:{plan['rr']:.1f}"
                + ("" if plan["ok"] else f"\n⛔ Se descartaría: {plan['reason']}")
            )
            lines.append(format_checks(mandatory, optional))
        elif data_ready(row):
            # Show the checklist for the side the EMAs lean towards, for transparency.
            lean = "LONG" if row["ema_fast"] > row["ema_slow"] else "SHORT"
            mandatory, optional, _ = side_checks(row, lean, ctx)
            lines.append(
                f"Sin señal. Chequeo {lean} "
                f"({sum(ok for _, ok in optional)}/{len(optional)}, mínimo {MIN_SCORE}):"
            )
            lines.append(format_checks(mandatory, optional))
        lines += [SEPARATOR] + format_market_view(ctx, row, float(row["close"]), news, now_utc)
    else:
        lines.append("Datos insuficientes para analizar.")
    lines.append("\nLa ejecución manual no envía órdenes.")
    telegram("\n".join(lines))


def main():
    if not KEY or not SECRET:
        raise RuntimeError("Faltan ALPACA_KEY / ALPACA_SECRET.")

    now_utc = pd.Timestamp.now(tz="UTC")
    now_ny = now_utc.tz_convert(NY)
    state = load_state()
    today = now_ny.strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "count": 0, "last_bar": ""}

    # Manual workflow runs are diagnostics only; they never place orders.
    if MANUAL:
        run_diagnostics(now_utc)
        save_state(state)
        return

    clock = get_clock()
    if not clock.get("is_open", False):
        print("Mercado cerrado según Alpaca; no se opera.")
        save_state(state)
        return

    if now_ny.time() >= FORCE_FLAT_TIME:
        close_position_end_of_day()
        save_state(state)
        return

    if not (ENTRY_START <= now_ny.time() < ENTRY_END):
        print(f"Fuera del horario de nuevas entradas: {now_ny.strftime('%H:%M')} NY.")
        save_state(state)
        return

    frame = get_bars()
    if frame.empty:
        print("No se recibieron velas.")
        save_state(state)
        return

    frame = frame.between_time("09:30", "15:59")
    # Keep only completed 5-minute candles to avoid acting on a still-forming bar.
    completed = frame[frame.index + pd.Timedelta(minutes=5) <= now_utc]
    if len(completed) < 100:
        print(f"Datos insuficientes: {len(completed)} velas cerradas.")
        save_state(state)
        return

    # Guard against stale data.
    latest_bar_time = completed.index[-1]
    if now_ny - latest_bar_time > pd.Timedelta(minutes=15):
        print("Última vela demasiado antigua; no se opera.")
        save_state(state)
        return

    enriched = add_indicators(completed)
    market = get_market_context(now_utc)
    candidates = scan(enriched, market, state.get("last_bar", ""))
    if not candidates:
        last = enriched.iloc[-1]
        print(f"Sin señal. Cierre {last['close']:.2f}, RSI {last['rsi']:.1f}, ADX {last['adx']:.1f}, mercado {market}.")
        save_state(state)
        return

    candidate = candidates[-1]
    side, row = candidate["side"], candidate["row"]
    # From here on this bar is consumed, whatever happens next.
    state["last_bar"] = candidate["bar_iso"]

    signal_price = float(row["close"])
    atr = float(row["atr"])
    latest_price = float(enriched["close"].iloc[-1])
    max_drift = MAX_DRIFT_ATR * ATR_MULT * atr

    if abs(latest_price - signal_price) > max_drift:
        telegram(
            "⏰ Señal descartada por movimiento tardío\n"
            f"{SYMBOL} {side}; señal {signal_price:.2f}, último cierre {latest_price:.2f}.\n"
            "No se envió ninguna orden."
        )
        save_state(state)
        return

    if state.get("count", 0) >= MAX_TRADES_DAY:
        print("Límite diario de operaciones alcanzado.")
        save_state(state)
        return

    busy, reason = has_position_or_open_order()
    if busy:
        print(reason)
        save_state(state)
        return

    if side == "SHORT":
        allowed, reason = can_short()
        if not allowed:
            telegram(f"⚠️ Señal SHORT de {SYMBOL} omitida: {reason}")
            save_state(state)
            return

    plan = make_plan(side, latest_price, atr, candidate["ctx"]["levels"])
    if not plan["ok"]:
        print(f"Señal {side} descartada: {plan['reason']}")
        save_state(state)
        return

    equity, buying_power = get_account()
    qty, risk_dollars = position_size(equity, buying_power, latest_price, plan["risk"])
    if qty < 1:
        telegram(
            f"⚠️ Señal {SYMBOL} omitida: el tamaño calculado es menor a 1 acción.\n"
            f"Equity ${equity:.2f}; riesgo objetivo ${risk_dollars:.2f}; "
            f"distancia stop ${plan['risk']:.2f}."
        )
        save_state(state)
        return

    news = get_news(now_utc)
    signal_text = format_signal(candidate, plan, qty, equity, news, now_utc)
    # Persist before the (long) approval wait so a crash cannot re-ask for the same bar.
    save_state(state)

    decision = ask_approval(signal_text)
    if decision != "approve":
        print(f"Decisión: {decision}. No se opera.")
        return

    # Re-validate everything after the human delay: market, exposure, live price.
    if not get_clock().get("is_open", False):
        telegram("⚠️ El mercado cerró mientras esperaba la aprobación. No se envió la orden.")
        return
    busy, reason = has_position_or_open_order()
    if busy:
        telegram(f"⚠️ {reason} No se envió la orden.")
        return
    entry_now = get_latest_price()
    if abs(entry_now - signal_price) > max_drift:
        telegram(
            f"⚠️ El precio se movió demasiado durante la aprobación ({signal_price:.2f} → {entry_now:.2f}).\n"
            "Orden cancelada por seguridad."
        )
        return

    # Stop and target stay at the planned prices (they are tied to levels);
    # check they still make sense from the live price and resize to the real risk.
    stop, target = plan["stop"], plan["target"]
    sign = 1 if side == "LONG" else -1
    risk_now = sign * (entry_now - stop)
    reward_now = sign * (target - entry_now)
    if risk_now < 0.02 or reward_now < 0.02 or reward_now / risk_now < MIN_RR_AFTER_APPROVAL:
        telegram(
            f"⚠️ Con el precio actual (${entry_now:.2f}) el stop ${stop:.2f} y el target "
            f"${target:.2f} ya no dan un riesgo/beneficio aceptable. Orden cancelada."
        )
        return
    equity, buying_power = get_account()
    qty, _ = position_size(equity, buying_power, entry_now, risk_now)
    if qty < 1:
        telegram("⚠️ Tamaño de posición menor a 1 acción tras la aprobación. No se operó.")
        return

    client_order_id = f"bot-{SYMBOL}-{candidate['time']:%Y%m%d%H%M}-{secrets.token_hex(3)}"
    try:
        order = send_order(side, qty, stop, target, client_order_id)
    except requests.HTTPError as exc:
        body = exc.response.text[:300] if exc.response is not None else str(exc)
        telegram(f"❌ Alpaca rechazó la orden: {body}")
        raise

    state["count"] = state.get("count", 0) + 1
    save_state(state)

    order = wait_for_fill(order["id"])
    fill = order.get("filled_avg_price")
    telegram(
        f"🚀 Orden enviada a Alpaca PAPER — {'🟢 LONG' if side == 'LONG' else '🔴 SHORT'} {SYMBOL}\n"
        f"Estado: {order.get('status')}\n"
        f"Cantidad: {qty}\n"
        f"Precio de ejecución: {('$' + format(float(fill), '.2f')) if fill else 'pendiente'}\n"
        f"🛑 Stop loss: ${stop:.2f}\n"
        f"🎯 Take profit: ${target:.2f}\n"
        f"Operaciones hoy: {state['count']}/{MAX_TRADES_DAY}\n"
        f"ID: {client_order_id}"
    )


if __name__ == "__main__":
    main()
