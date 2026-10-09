"""
Backtest ORB (Opening Range Breakout) + VWAP para NVDA, long y short.
Datos: Alpaca (velas de 5 min). Resultado: reporte.md + trades.csv + resumen a Telegram.

Uso:
  python backtest_orb.py            -> datos reales de Alpaca
  python backtest_orb.py --demo     -> datos simulados (solo para probar que el código corre)

Variables de entorno (GitHub Secrets):
  ALPACA_API_KEY, ALPACA_SECRET_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import sys
import itertools
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG
SIMBOLO = "NVDA"
FILTRO = "QQQ"                 # filtro de tendencia del mercado
ANIOS = 2
CAPITAL_INICIAL = 10_000.0
RIESGO_POR_TRADE = 0.005       # 0,5% de la cuenta
PERDIDA_DIARIA_MAX = 0.02      # 2% de la cuenta
MAX_TRADES_DIA = 2
SLIPPAGE = 0.0002              # 0,02% por lado
ANTI_DESLIZ_ATR = 0.3          # no entrar si el precio se alejó > 0,3 ATR
SALIDA_FORZADA = "15:50"       # se sale al cierre de la vela 15:50-15:55 (hora NY)
TRAIN_FRAC = 0.70

# Parámetros base (los que se evalúan "a ciegas" en el test)
BASE = {"or_min": 15, "atr_mult": 1.0, "rr": 2.0}
# Grilla para ver robustez (solo se optimiza sobre TRAIN)
GRILLA = {"or_min": [5, 15, 30], "atr_mult": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 3.0]}

# Compuertas
MIN_TRADES = 100
MIN_PF = 1.3
MAX_DD = 0.15
MIN_RATIO_OOS = 0.5


# ------------------------------------------------------------------ DATOS
def bajar_datos_alpaca():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    fin = datetime.now(timezone.utc) - timedelta(minutes=20)  # plan gratis: SIP con 15 min de demora
    inicio = fin - timedelta(days=365 * ANIOS)

    def pedir(feed):
        req = StockBarsRequest(
            symbol_or_symbols=[SIMBOLO, FILTRO],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=inicio, end=fin, adjustment=Adjustment.ALL, feed=feed,
        )
        return client.get_stock_bars(req).df

    try:
        df = pedir("sip")
        feed = "SIP (completo)"
    except Exception as e:
        print(f"SIP no disponible ({e}); uso IEX")
        df = pedir("iex")
        feed = "IEX (parcial)"

    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    out = {}
    for s in (SIMBOLO, FILTRO):
        d = df[df["symbol"] == s].set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        out[s] = d.sort_index()
    return out, feed


def datos_demo():
    """Random walk intradía. NO tiene ventaja real: solo sirve para probar el código."""
    rng = np.random.default_rng(42)
    dias = pd.bdate_range("2024-10-01", periods=500, tz="America/New_York")
    out = {}
    for s, p0 in ((SIMBOLO, 120.0), (FILTRO, 480.0)):
        filas, p = [], p0
        for d in dias:
            p *= np.exp(rng.normal(0, 0.01))  # gap nocturno
            for k in range(78):
                ts = d + pd.Timedelta(hours=9, minutes=30 + 5 * k)
                r = rng.normal(0, 0.003)
                o, c = p, p * np.exp(r)
                h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
                l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
                filas.append((ts, o, h, l, c, rng.integers(5e4, 3e5)))
                p = c
        out[s] = pd.DataFrame(filas, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
    return out, "DEMO (simulado)"


def preparar(df):
    """Sesión regular, VWAP diario, ATR(14), media de volumen 20."""
    df = df.between_time("09:30", "15:55").copy()
    df["fecha"] = df.index.date
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).groupby(df["fecha"]).cumsum()
    vv = df["volume"].groupby(df["fecha"]).cumsum().replace(0, np.nan)
    df["vwap"] = pv / vv
    prev_c = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


# ------------------------------------------------------------------ BACKTEST
def backtest(nv, qq, or_min, atr_mult, rr, fechas):
    """Simula día por día. Señal al cierre de la vela i, entrada a la apertura de i+1."""
    equity = CAPITAL_INICIAL
    trades, curva = [], []
    n_or = or_min // 5
    qq_arriba = (qq["close"] > qq["vwap"])

    for fecha in fechas:
        d = nv[nv["fecha"] == fecha]
        if len(d) < n_or + 3:
            continue
        o_hi, o_lo = d["high"].iloc[:n_or].max(), d["low"].iloc[:n_or].min()
        idx = d.index
        q_dia = qq_arriba.reindex(idx)
        eq_ini_dia, n_trades, i = equity, 0, n_or

        while i < len(d) - 1 and n_trades < MAX_TRADES_DIA:
            if equity <= eq_ini_dia * (1 - PERDIDA_DIARIA_MAX):
                break
            if idx[i].strftime("%H:%M") >= SALIDA_FORZADA:
                break
            row = d.iloc[i]
            atr = row["atr"]
            if np.isnan(atr) or np.isnan(row["vol_ma"]) or atr <= 0:
                i += 1
                continue
            vol_ok = row["volume"] > row["vol_ma"]
            q = q_dia.iloc[i]
            lado = 0
            if row["close"] > o_hi and row["close"] > row["vwap"] and vol_ok and q == True:  # noqa: E712
                lado = 1
            elif row["close"] < o_lo and row["close"] < row["vwap"] and vol_ok and q == False:  # noqa: E712
                lado = -1
            if lado == 0:
                i += 1
                continue

            # entrada en la vela siguiente
            sig = row["close"]
            nxt = d.iloc[i + 1]
            entrada_raw = nxt["open"]
            if lado * (entrada_raw - sig) > ANTI_DESLIZ_ATR * atr:  # regla anti-deslizamiento
                i += 1
                continue
            entrada = entrada_raw * (1 + lado * SLIPPAGE)
            if lado == 1:
                stop = max(o_lo, entrada - atr_mult * atr)
            else:
                stop = min(o_hi, entrada + atr_mult * atr)
            r = abs(entrada - stop)
            if r <= 0:
                i += 1
                continue
            tp = entrada + lado * rr * r
            qty = min((equity * RIESGO_POR_TRADE) / r, equity / entrada)  # sin apalancamiento

            # gestión del trade
            salida, motivo, j = None, None, i + 1
            while j < len(d):
                b = d.iloc[j]
                if lado == 1:
                    if b["low"] <= stop:          # stop primero (conservador)
                        salida, motivo = min(stop, b["open"]), "SL"
                    elif b["high"] >= tp:
                        salida, motivo = max(tp, b["open"]), "TP"
                else:
                    if b["high"] >= stop:
                        salida, motivo = max(stop, b["open"]), "SL"
                    elif b["low"] <= tp:
                        salida, motivo = min(tp, b["open"]), "TP"
                if salida is None and (idx[j].strftime("%H:%M") >= SALIDA_FORZADA or j == len(d) - 1):
                    salida, motivo = b["close"], "CIERRE"
                if salida is not None:
                    break
                j += 1

            salida *= (1 - lado * SLIPPAGE)
            pnl = lado * (salida - entrada) * qty
            equity += pnl
            trades.append({
                "fecha": fecha, "hora": idx[i + 1].strftime("%H:%M"), "lado": "LONG" if lado == 1 else "SHORT",
                "entrada": round(entrada, 2), "stop": round(stop, 2), "tp": round(tp, 2),
                "salida": round(salida, 2), "motivo": motivo, "qty": round(qty, 2),
                "pnl": round(pnl, 2), "R": round(lado * (salida - entrada) / r, 2), "equity": round(equity, 2),
            })
            n_trades += 1
            i = j + 1
        curva.append((fecha, equity))
    return pd.DataFrame(trades), pd.DataFrame(curva, columns=["fecha", "equity"])


def metricas(tr, curva):
    if tr.empty:
        return {"trades": 0, "win_rate": 0, "pf": 0, "exp_R": 0, "retorno": 0, "max_dd": 0, "sharpe": 0}
    g, p = tr.loc[tr.pnl > 0, "pnl"].sum(), -tr.loc[tr.pnl < 0, "pnl"].sum()
    eq = curva["equity"]
    dd = (eq / eq.cummax() - 1).min()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    return {
        "trades": len(tr), "win_rate": (tr.pnl > 0).mean(), "pf": g / p if p > 0 else np.inf,
        "exp_R": tr["R"].mean(), "retorno": eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) else 0,
        "max_dd": -dd, "sharpe": sharpe,
    }


def fmt(m):
    return (f"Trades {m['trades']} | Win {m['win_rate']:.0%} | PF {m['pf']:.2f} | "
            f"Exp {m['exp_R']:+.2f}R | Ret {m['retorno']:+.1%} | MaxDD {m['max_dd']:.1%} | Sharpe {m['sharpe']:.2f}")


# ------------------------------------------------------------------ TELEGRAM
def telegram(texto):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    import requests
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": texto[:4000]}, timeout=20)
    except Exception as e:
        print("No se pudo enviar a Telegram:", e)


# ------------------------------------------------------------------ MAIN
def main():
    demo = "--demo" in sys.argv
    datos, feed = datos_demo() if demo else bajar_datos_alpaca()
    nv, qq = preparar(datos[SIMBOLO]), preparar(datos[FILTRO])

    fechas = sorted(nv["fecha"].unique())
    corte = int(len(fechas) * TRAIN_FRAC)
    f_train, f_test = fechas[:corte], fechas[corte:]

    # 1) Parámetros base, sin optimizar
    tr_is, c_is = backtest(nv, qq, fechas=f_train, **BASE)
    tr_oos, c_oos = backtest(nv, qq, fechas=f_test, **BASE)
    m_is, m_oos = metricas(tr_is, c_is), metricas(tr_oos, c_oos)

    # 2) Grilla solo en TRAIN (robustez)
    filas = []
    for o, a, r in itertools.product(GRILLA["or_min"], GRILLA["atr_mult"], GRILLA["rr"]):
        t, c = backtest(nv, qq, o, a, r, f_train)
        m = metricas(t, c)
        filas.append({"or_min": o, "atr_mult": a, "rr": r, **m})
    grilla = pd.DataFrame(filas).sort_values("pf", ascending=False)
    pct_positivos = (grilla["exp_R"] > 0).mean()

    # 3) Buy & hold en el período de test
    px = nv[nv["fecha"].isin(f_test)]["close"]
    bh = px.iloc[-1] / px.iloc[0] - 1 if len(px) else 0

    # 4) Compuertas (sobre OOS)
    trades_tot = m_is["trades"] + m_oos["trades"]
    ratio = (m_oos["exp_R"] / m_is["exp_R"]) if m_is["exp_R"] > 0 else 0
    comp = {
        f"≥{MIN_TRADES} trades en total": trades_tot >= MIN_TRADES,
        f"PF test > {MIN_PF}": m_oos["pf"] > MIN_PF,
        f"MaxDD test < {MAX_DD:.0%}": m_oos["max_dd"] < MAX_DD,
        f"Test ≥ {MIN_RATIO_OOS:.0%} del train (expectativa)": ratio >= MIN_RATIO_OOS,
        "≥60% de la grilla con expectativa positiva": pct_positivos >= 0.6,
    }
    aprobado = all(comp.values())

    # Reporte
    lineas = [
        f"# Backtest ORB {SIMBOLO} — {datetime.now().strftime('%Y-%m-%d')}",
        f"Datos: {feed} | {fechas[0]} a {fechas[-1]} | Train {len(f_train)} días / Test {len(f_test)} días",
        "",
        f"## Parámetros base {BASE}",
        f"- TRAIN: {fmt(m_is)}",
        f"- TEST:  {fmt(m_oos)}",
        f"- Buy & hold {SIMBOLO} en el test: {bh:+.1%}",
        "",
        "## Compuertas",
        *[f"- {'✅' if v else '❌'} {k}" for k, v in comp.items()],
        f"\n**Veredicto: {'PASA → seguir a paper trading' if aprobado else 'NO PASA → iterar o descartar'}**",
        "",
        "## Grilla en TRAIN (top 10 por PF)",
        grilla.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"),
        "",
        f"Combinaciones con expectativa positiva: {pct_positivos:.0%}",
    ]
    reporte = "\n".join(lineas)
    open("reporte.md", "w", encoding="utf-8").write(reporte)
    pd.concat([tr_is.assign(set="train"), tr_oos.assign(set="test")]).to_csv("trades.csv", index=False)
    print(reporte)

    resumen = "\n".join([
        f"📊 Backtest ORB {SIMBOLO} ({feed})",
        f"TRAIN: {fmt(m_is)}",
        f"TEST: {fmt(m_oos)}",
        f"Buy&hold test: {bh:+.1%}",
        *[f"{'✅' if v else '❌'} {k}" for k, v in comp.items()],
        f"Veredicto: {'PASA' if aprobado else 'NO PASA'}",
    ])
    telegram(resumen)


if __name__ == "__main__":
    main()
