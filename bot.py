import os
import json
import math
import time
import secrets
from datetime import time as dtime

import pandas as pd
import requests

# SAFETY: this version is hard-wired to Alpaca PAPER trading.
TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
SYMBOL = "NVDA"
NY = "America/New_York"
STATE_FILE = "estado.json"

# Risk controls
RISK_PCT = 0.5                 # risk budget as % of account equity per trade
ATR_MULT = 1.5                 # stop distance = ATR * this multiplier
RR = 2.0                       # take-profit distance = stop distance * this ratio
ADX_MIN = 20
MAX_TRADES_DAY = 3
MAX_NOTIONAL_PCT = 20.0        # cap exposure to 20% of equity per position
MAX_DRIFT_ATR = 0.5             # reject if price moved too far from signal close
MAX_OPEN_POSITIONS = 1
ENTRY_START = dtime(9, 45)
# Relative to Alpaca's real close, so half-day sessions (13:00 close) are handled.
ENTRY_STOP_BEFORE_CLOSE = pd.Timedelta(minutes=40)   # 15:20 on a normal day
FORCE_FLAT_BEFORE_CLOSE = pd.Timedelta(minutes=10)   # 15:50 on a normal day
ERROR_REPEAT_SECONDS = 3600    # don't resend the same error alert more than once an hour

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


def get_bars():
    start = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{DATA_URL}/v2/stocks/{SYMBOL}/bars"
    params = {
        "timeframe": "5Min",
        "start": start,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
    }
    bars = []
    while True:
        response = SESSION.get(url, headers=HEADERS, params=params, timeout=30)
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
    frame["t"] = pd.to_datetime(frame["t"], utc=True).dt.tz_convert(NY)
    frame = frame.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame = frame.set_index("t")[["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def rma(series, period):
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


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
    df["sweep_bull_recent"] = df["sweep_bull"].rolling(6, min_periods=1).max().fillna(False).astype(bool)
    df["sweep_bear_recent"] = df["sweep_bear"].rolling(6, min_periods=1).max().fillna(False).astype(bool)
    df["fvg_bull_recent"] = df["fvg_bull"].rolling(6, min_periods=1).max().fillna(False).astype(bool)
    df["fvg_bear_recent"] = df["fvg_bear"].rolling(6, min_periods=1).max().fillna(False).astype(bool)

    return df


def evaluate(row):
    # Original indicator filters are retained; structure plus FVG/sweep add confluence.
    base = (
        pd.notna(row["adx"])
        and row["adx"] > ADX_MIN
        and bool(row["vol_ok"])
        and pd.notna(row["atr"])
        and row["atr"] > 0
        and pd.notna(row["vwap"])
        and pd.notna(row["rsi"])
        and pd.notna(row["rsi_prev"])
    )

    long_signal = (
        base
        and row["ema_fast"] > row["ema_slow"]
        and row["close"] > row["vwap"]
        and row["rsi_prev"] <= 45 < row["rsi"]
        and row["macd_hist"] > 0
        and bool(row["structure_up_recent"])
        and (bool(row["fvg_bull_recent"]) or bool(row["sweep_bull_recent"]))
        and row["market_bias"] == 1
    )
    short_signal = (
        base
        and row["ema_fast"] < row["ema_slow"]
        and row["close"] < row["vwap"]
        and row["rsi_prev"] >= 55 > row["rsi"]
        and row["macd_hist"] < 0
        and bool(row["structure_down_recent"])
        and (bool(row["fvg_bear_recent"]) or bool(row["sweep_bear_recent"]))
        and row["market_bias"] == -1
    )
    if long_signal:
        return "LONG"
    if short_signal:
        return "SHORT"
    return None


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"date": "", "count": 0, "last_bar": ""}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


# Keys that must survive the daily reset (an open trade, the last error alert).
PERSISTENT_KEYS = ("open_trade", "last_error", "last_error_at")


def new_day_state(today, old_state):
    state = {"date": today, "count": 0, "last_bar": ""}
    for key in PERSISTENT_KEYS:
        if key in old_state:
            state[key] = old_state[key]
    return state


def money(value):
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


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


def ask_approval(signal_text, timeout_seconds=180):
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
    message = telegram_api("sendMessage", {
        "chat_id": TG_CHAT,
        "text": signal_text + "\n\n⏳ Esperando tu decisión. Si no respondés en 3 minutos, no se opera.",
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
        "text": signal_text + "\n\n⌛ Señal vencida: no se recibió aprobación en 3 minutos. No se operó.",
        "reply_markup": {"inline_keyboard": []},
    })
    return "timeout"


def send_order(side, qty, stop, take_profit):
    payload = {
        "symbol": SYMBOL,
        "qty": str(qty),
        "side": "buy" if side == "LONG" else "sell",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{take_profit:.2f}"},
        "stop_loss": {"stop_price": f"{stop:.2f}"},
    }
    # The endpoint is hard-coded to paper-api.alpaca.markets above.
    return api("POST", "/v2/orders", payload=payload)


def has_position_or_open_order():
    positions = api("GET", "/v2/positions")
    if any(p.get("symbol") == SYMBOL for p in positions):
        return True, "Ya existe una posición abierta en NVDA."
    orders = api("GET", "/v2/orders", params={"status": "open", "symbols": SYMBOL, "limit": 100})
    if orders:
        return True, "Ya hay una orden abierta para NVDA."
    return False, ""


def close_position_end_of_day(state):
    orders = api("GET", "/v2/orders", params={"status": "open", "symbols": SYMBOL, "limit": 100})
    for order in orders or []:
        api("DELETE", f"/v2/orders/{order['id']}")
    position = api("GET", f"/v2/positions/{SYMBOL}", allow_404=True)
    if position:
        exit_order = api("DELETE", f"/v2/positions/{SYMBOL}")
        # Remember the closing order so the next run can report the result.
        if state.get("open_trade") and exit_order:
            state["open_trade"]["exit_order_id"] = exit_order.get("id")
        telegram("🧹 Cierre de fin de día solicitado para NVDA en PAPER. En unos minutos te llega el resultado.")
    else:
        print("Fin de día: no hay posición NVDA abierta.")


def check_open_trade(state):
    """Report the result of the bot's trade once its stop, take profit or EOD close fills."""
    trade = state.get("open_trade")
    if not trade:
        return

    order = api("GET", f"/v2/orders/{trade['order_id']}", params={"nested": "true"})
    if float(order.get("filled_qty") or 0) == 0:
        if order.get("status") in ("canceled", "expired", "rejected"):
            telegram(f"ℹ️ La orden NVDA {trade['side']} no se ejecutó (estado: {order['status']}).")
            state.pop("open_trade")
        return

    side = trade["side"]
    qty = float(order["filled_qty"])
    entry = float(order["filled_avg_price"])
    exit_price, reason = None, None
    for leg in order.get("legs") or []:
        if leg.get("status") == "filled":
            exit_price = float(leg["filled_avg_price"])
            reason = "TAKE PROFIT" if leg.get("type") == "limit" else "STOP LOSS"
            break
    if exit_price is None and trade.get("exit_order_id"):
        exit_order = api("GET", f"/v2/orders/{trade['exit_order_id']}")
        if exit_order.get("status") == "filled":
            exit_price = float(exit_order["filled_avg_price"])
            reason = "CIERRE DE FIN DE DÍA"

    if exit_price is None:
        if api("GET", f"/v2/positions/{SYMBOL}", allow_404=True):
            return  # still open
        telegram(
            f"ℹ️ La posición NVDA {side} se cerró fuera del bot.\n"
            "Revisá el resultado en Alpaca."
        )
        state.setdefault("closed_today", []).append(
            {"side": side, "qty": qty, "reason": "CERRADA FUERA DEL BOT", "pnl": None}
        )
        state.pop("open_trade")
        return

    pnl = (exit_price - entry) * qty * (1 if side == "LONG" else -1)
    icon = {"TAKE PROFIT": "🎯", "STOP LOSS": "🛑"}.get(reason, "🧹")
    telegram(
        f"{icon} {reason}: NVDA {side} cerrada\n"
        f"Cantidad: {qty:g}\n"
        f"Entrada: {entry:.2f} → Salida: {exit_price:.2f}\n"
        f"Resultado: {money(pnl)}"
    )
    state.setdefault("closed_today", []).append(
        {"side": side, "qty": qty, "reason": reason, "pnl": round(pnl, 2)}
    )
    state.pop("open_trade")


def send_open_message(close_ny):
    account = api("GET", "/v2/account")
    half_day = " (media jornada)" if close_ny.time() < dtime(16, 0) else ""
    telegram(
        "✅ Bot activo: mercado abierto\n"
        f"Equity: ${float(account['equity']):,.2f}\n"
        f"Cierre de hoy: {close_ny.strftime('%H:%M')} NY{half_day}\n"
        "Si no hay señales, no vas a recibir más mensajes hasta el resumen del cierre."
    )


def send_daily_summary(state):
    account = api("GET", "/v2/account")
    equity = float(account["equity"])
    last_equity = float(account.get("last_equity") or equity)
    closed = state.get("closed_today", [])

    lines = [
        f"📊 Resumen del día {state['date']} (PAPER)",
        f"Operaciones enviadas hoy: {state.get('count', 0)}",
    ]
    if closed:
        lines.append("Operaciones cerradas:")
        for trade in closed:
            result = money(trade["pnl"]) if trade.get("pnl") is not None else "sin dato"
            lines.append(f"• {trade['side']} x{trade['qty']:g}: {trade['reason']} {result}")
    else:
        lines.append("Operaciones cerradas: ninguna")
    lines.append(f"Resultado del día (cuenta Alpaca): {money(equity - last_equity)}")
    lines.append(f"Equity: ${equity:,.2f}")
    position = api("GET", f"/v2/positions/{SYMBOL}", allow_404=True)
    if position:
        lines.append(f"⚠️ Quedó una posición NVDA abierta: {position.get('qty')} acciones. Revisala en Alpaca.")
    telegram("\n".join(lines))


def redact(text):
    for secret in (TG_TOKEN, KEY, SECRET):
        if secret:
            text = text.replace(secret, "***")
    return text


def report_error(exc):
    """Alert by Telegram when a run crashes, without repeating the same error every 5 minutes."""
    error = redact(f"{type(exc).__name__}: {exc}")[:800]
    state = load_state()
    now = time.time()
    if state.get("last_error") == error and now - state.get("last_error_at", 0) < ERROR_REPEAT_SECONDS:
        print("Mismo error que la ejecución anterior; no se reenvía a Telegram.")
        return
    try:
        telegram(f"⚠️ El bot falló\n{error}\n\nRevisá el log en GitHub Actions.")
    except Exception as tg_exc:
        print("No se pudo avisar el error por Telegram:", redact(str(tg_exc)))
        return
    state["last_error"] = error
    state["last_error_at"] = now
    save_state(state)


def clear_error():
    state = load_state()
    if "last_error" not in state:
        return
    state.pop("last_error")
    state.pop("last_error_at", None)
    save_state(state)
    try:
        telegram("✅ El bot volvió a funcionar normalmente.")
    except Exception as tg_exc:
        print("No se pudo avisar la recuperación por Telegram:", redact(str(tg_exc)))


def main():
    if not KEY or not SECRET:
        raise RuntimeError("Faltan ALPACA_KEY o ALPACA_SECRET en los secrets de GitHub.")

    now_utc = pd.Timestamp.now(tz="UTC")
    now_ny = now_utc.tz_convert(NY)
    state = load_state()
    today = now_ny.strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = new_day_state(today, state)

    # Manual workflow runs are diagnostics only; they never place orders.
    if MANUAL:
        account_equity, _ = get_account()
        clock = get_clock()
        telegram(
            "🧪 Diagnóstico PAPER solamente\n"
            f"Cuenta equity: ${account_equity:.2f}\n"
            f"Mercado abierto según Alpaca: {clock.get('is_open')}\n"
            f"Hora Nueva York: {now_ny.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "La ejecución manual no envía órdenes."
        )
        save_state(state)
        return

    check_open_trade(state)
    save_state(state)

    clock = get_clock()
    if not clock.get("is_open", False):
        if state.get("session_seen") and not state.get("summary_sent"):
            send_daily_summary(state)
            state["summary_sent"] = True
        print("Mercado cerrado según Alpaca; no se opera.")
        save_state(state)
        return

    close_ny = pd.Timestamp(clock["next_close"]).tz_convert(NY)
    entry_end = (close_ny - ENTRY_STOP_BEFORE_CLOSE).time()

    if not state.get("session_seen"):
        send_open_message(close_ny)
        state["session_seen"] = True
        save_state(state)

    if now_ny >= close_ny - FORCE_FLAT_BEFORE_CLOSE:
        close_position_end_of_day(state)
        save_state(state)
        return

    if not (ENTRY_START <= now_ny.time() < entry_end):
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
    candidates = []
    for bar_time, row in enriched.iloc[-3:].iterrows():
        if bar_time.time() < ENTRY_START or bar_time.time() >= entry_end:
            continue
        bar_iso = bar_time.tz_convert("UTC").isoformat()
        if bar_iso <= state.get("last_bar", ""):
            continue
        side = evaluate(row)
        if side:
            candidates.append((bar_time, row, side, bar_iso))

    if not candidates:
        save_state(state)
        return

    bar_time, row, side, bar_iso = candidates[-1]
    signal_price = float(row["close"])
    atr = float(row["atr"])
    stop_distance = atr * ATR_MULT
    latest_price = float(enriched["close"].iloc[-1])

    if abs(latest_price - signal_price) > MAX_DRIFT_ATR * stop_distance:
        state["last_bar"] = bar_iso
        telegram(
            "⏰ Señal descartada por movimiento tardío\n"
            f"NVDA {side}; señal {signal_price:.2f}, último cierre {latest_price:.2f}.\n"
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

    equity, buying_power = get_account()
    risk_dollars = equity * (RISK_PCT / 100.0)
    max_notional = equity * (MAX_NOTIONAL_PCT / 100.0)
    qty_by_risk = math.floor(risk_dollars / stop_distance)
    qty_by_notional = math.floor(max_notional / signal_price)
    qty_by_buying_power = math.floor(buying_power / signal_price)
    qty = min(qty_by_risk, qty_by_notional, qty_by_buying_power)

    if qty < 1:
        telegram(
            "⚠️ Señal NVDA omitida: el tamaño calculado es menor a 1 acción.\n"
            f"Equity ${equity:.2f}; riesgo objetivo ${risk_dollars:.2f}; "
            f"distancia stop ${stop_distance:.2f}."
        )
        state["last_bar"] = bar_iso
        save_state(state)
        return


    if side == "LONG":
        stop_price = round(signal_price - stop_distance, 2)
        tp_price = round(signal_price + stop_distance * RR, 2)
    else:
        stop_price = round(signal_price + stop_distance, 2)
        tp_price = round(signal_price - stop_distance * RR, 2)

    hora = (bar_time + pd.Timedelta(minutes=5)).strftime("%H:%M")
    titulo = "COMPRAR (LONG)" if side == "LONG" else "VENDER EN CORTO (SHORT)"
    signal_text = (
        f"📈 {titulo} NVDA (PAPER)\n"
        f"Señal de la vela de las {hora} (hora NY)\n"
        f"Cantidad: {qty}\n"
        f"Precio señal: {signal_price:.2f} (último cierre {latest_price:.2f})\n"
        f"STOP LOSS: {stop_price:.2f}\n"
        f"TAKE PROFIT: {tp_price:.2f}"
    )

    # Mark the bar as handled before asking, so the same signal is never re-sent.
    state["last_bar"] = bar_iso
    save_state(state)

    decision = ask_approval(signal_text)
    if decision != "approve":
        print(f"Operación no aprobada: {decision}.")
        return

    order = send_order(side, qty, stop_price, tp_price)
    state["count"] = state.get("count", 0) + 1
    state["open_trade"] = {"order_id": order["id"], "side": side}
    save_state(state)
    telegram(
        f"✅ Orden enviada a Alpaca PAPER\n"
        f"ID: {order.get('id')}\nEstado: {order.get('status')}"
    )


def run():
    try:
        main()
    except Exception as exc:
        report_error(exc)
        raise
    clear_error()


if __name__ == "__main__":
    run()
