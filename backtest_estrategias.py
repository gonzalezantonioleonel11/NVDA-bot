"""
Comparación de 7 estrategias de day trading para NVDA (velas de 5 min, long y short).

Todas usan las mismas reglas de ejecución y de riesgo. Los parámetros base se fijan de
antemano (los de la literatura o los habituales), se miden en TRAIN (70% de los días) y se
validan en TEST (30% final, que no se usa para elegir nada). Se aplican las mismas compuertas
que en backtest_orb.py y el ranking llega por Telegram.

Uso:
  python backtest_estrategias.py          -> datos reales de Alpaca
  python backtest_estrategias.py --demo   -> datos simulados (solo para probar que el código corre)

Variables de entorno (GitHub Secrets):
  ALPACA_API_KEY, ALPACA_SECRET_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import sys
import time
import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG
SIMBOLO = "NVDA"
FILTRO = "QQQ"                  # filtro de mercado
DESDE = "2021-10-01"            # ~5 años: incluye el 2022 bajista y el rally 2023-2025
CAPITAL_INICIAL = 10_000.0
RIESGO_POR_TRADE = 0.005        # 0,5% de la cuenta por trade (sin apalancamiento)
PERDIDA_DIARIA_MAX = 0.02       # se deja de operar el día si se pierde 2%
SLIPPAGE = 0.0002               # 0,02% por lado
ANTI_DESLIZ_ATR = 0.3           # no entrar si el precio se alejó > 0,3 ATR de la señal
MIN_STOP_ATR = 0.5              # el stop queda como mínimo a 0,5 ATR (5 min) de la entrada
TRAIN_FRAC = 0.70

NB = 78                         # velas de 5 min de la sesión regular (9:30-16:00 NY)
K_SALIDA = 76                   # salida forzada al cierre de la vela 15:50-15:55
K_ULTIMA_SENAL = 65             # última señal: cierre de la vela 14:55-15:00
KIDX = np.arange(NB)[None, :]

# Compuertas (las mismas que backtest_orb.py)
MIN_TRADES = 100
MIN_PF = 1.3
MAX_DD = 0.15
MIN_RATIO_OOS = 0.5
MIN_GRILLA = 0.6


# ------------------------------------------------------------------ DATOS
def bajar_datos_alpaca():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    inicio = datetime.fromisoformat(DESDE).replace(tzinfo=timezone.utc)
    fin = datetime.now(timezone.utc) - timedelta(minutes=20)  # plan gratis: SIP con 15 min de demora

    def pedir(feed):
        req = StockBarsRequest(
            symbol_or_symbols=[SIMBOLO, FILTRO],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=inicio, end=fin, adjustment=Adjustment.ALL, feed=feed,
        )
        return client.get_stock_bars(req).df

    try:
        df, feed = pedir("sip"), "SIP"
    except Exception as e:
        print(f"SIP no disponible ({e}); uso IEX")
        df, feed = pedir("iex"), "IEX (parcial)"

    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    cols = ["open", "high", "low", "close", "volume"]
    return {s: df[df["symbol"] == s].set_index("timestamp")[cols].sort_index() for s in (SIMBOLO, FILTRO)}, feed


def datos_demo(dias=1260, ventaja=0.0, seed=7, pasos=30):
    """Paseo al azar intradía (martingala): NO tiene ventaja real, solo sirve para probar el código.
    Cada vela de 5 min se arma con `pasos` movimientos más chicos, así máximo y mínimo son los del
    camino real. Con ventaja > 0 el resto del día sigue la dirección de la 1ª vela (para verificar
    que el motor detecta una ventaja cuando existe)."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range(DESDE, periods=dias)
    horas = (570 + 5 * np.arange(NB)).astype("timedelta64[m]")
    idx = pd.DatetimeIndex((fechas.values[:, None] + horas[None, :]).ravel()).tz_localize("America/New_York")
    s_m, s_i = 0.0025 / np.sqrt(pasos), 0.002 / np.sqrt(pasos)     # volatilidad por paso
    mercado = rng.normal(0, s_m, (dias, NB, pasos))
    forma_vol = 1 + 2 * np.exp(-np.arange(NB) / 6) + np.exp(-(NB - 1 - np.arange(NB)) / 6)
    out = {}
    for s, p0, beta in ((SIMBOLO, 30.0, 1.0), (FILTRO, 350.0, 0.6)):
        r = beta * mercado + rng.normal(0, s_i, (dias, NB, pasos))
        r -= 0.5 * ((beta * s_m) ** 2 + s_i ** 2)                     # sin deriva: martingala en precio
        if ventaja and s == SIMBOLO:
            r[:, 1:, :] += ventaja / pasos * np.sign(r[:, :1, :].sum(axis=2, keepdims=True))
        cum = np.cumsum(r.reshape(dias, NB * pasos), axis=1)
        gaps = rng.normal(0, 0.01, dias)
        aper, p = np.empty(dias), p0
        for d in range(dias):
            aper[d] = p * np.exp(gaps[d])
            p = aper[d] * np.exp(cum[d, -1])
        camino = (aper[:, None] * np.exp(cum)).reshape(dias, NB, pasos)
        c = camino[:, :, -1]
        o = np.concatenate([aper[:, None], c[:, :-1]], axis=1)
        h = np.maximum(camino.max(axis=2), o)
        l = np.minimum(camino.min(axis=2), o)
        v = 1e5 * forma_vol[None, :] * rng.lognormal(0, 0.4, c.shape)
        out[s] = pd.DataFrame({"open": o.ravel(), "high": h.ravel(), "low": l.ravel(),
                               "close": c.ravel(), "volume": v.ravel()}, index=idx)
    return out, "DEMO (simulado)"


def a_matrices(df):
    """Pasa las velas a matrices días x 78 (sesión regular). Las velas que faltan quedan en NaN."""
    t = df.index
    m = (t.hour * 60 + t.minute).to_numpy() - 570
    sel = (m >= 0) & (m < 5 * NB) & (m % 5 == 0)
    df, m = df[sel], m[sel] // 5
    dia = np.asarray(df.index.date)
    fechas = sorted(set(dia))
    pos = {f: i for i, f in enumerate(fechas)}
    fila = np.array([pos[f] for f in dia], dtype=int)
    M = {}
    for col in ("open", "high", "low", "close", "volume"):
        a = np.full((len(fechas), NB), np.nan)
        a[fila, m] = df[col].to_numpy(dtype=float)
        M[col] = a
    return fechas, M


def n_velas(C):
    ok = ~np.isnan(C)
    return np.where(ok.any(1), NB - np.argmax(ok[:, ::-1], axis=1), 0)


def rellenar(M):
    """Completa velas sueltas que falten dentro del día con el último precio (volumen 0)."""
    C = M["close"]
    nb = n_velas(C)
    falta = np.isnan(C) & (KIDX < nb[:, None])
    if falta.any():
        cf = pd.DataFrame(C).ffill(axis=1).to_numpy()
        for col in ("open", "high", "low", "close"):
            M[col][falta] = cf[falta]
        M["volume"][falta] = 0.0
    return nb


def vwap(M):
    """VWAP del día y su desvío (ponderado por volumen) hasta cada vela, inclusive."""
    tp = np.nan_to_num((M["high"] + M["low"] + M["close"]) / 3)
    v = np.nan_to_num(M["volume"])
    cv, cpv, cp2 = np.cumsum(v, 1), np.cumsum(tp * v, 1), np.cumsum(tp * tp * v, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vw = np.where(cv > 0, cpv / cv, M["close"])
        var = np.where(cv > 0, cp2 / cv - vw * vw, 0.0)
    return vw, np.sqrt(np.clip(var, 0, None))


def preparar(datos):
    f_nv, M = a_matrices(datos[SIMBOLO])
    nb = n_velas(M["close"])
    ok = ~np.isnan(M["close"][:, 0]) & (nb >= 40)   # días con apertura y al menos media sesión
    fechas = [f for f, b in zip(f_nv, ok) if b]
    M = {c: a[ok] for c, a in M.items()}
    nbar = rellenar(M)
    O, H, L, C, V = (M[c] for c in ("open", "high", "low", "close", "volume"))
    D = len(fechas)
    VW, SD = vwap(M)

    # Serie continua de velas (ATR de 5 min, media de volumen, EMAs): solo usa velas pasadas
    fi, ki = np.nonzero(~np.isnan(C))
    c1, h1, l1, o1 = C[fi, ki], H[fi, ki], L[fi, ki], O[fi, ki]
    pc = np.where(ki == 0, o1, np.r_[c1[0], c1[:-1]])        # sin contar el gap de la noche
    tr = np.maximum(h1 - l1, np.maximum(np.abs(h1 - pc), np.abs(l1 - pc)))

    def a_matriz(x):
        z = np.full(C.shape, np.nan)
        z[fi, ki] = x
        return z

    ATR5 = a_matriz(pd.Series(tr).rolling(14).mean().to_numpy())
    VMA = a_matriz(pd.Series(V[fi, ki]).rolling(20).mean().to_numpy())
    cache = {}

    def ema(n):
        if n not in cache:
            cache[n] = a_matriz(pd.Series(c1).ewm(span=n, adjust=False).mean().to_numpy())
        return cache[n]

    # Datos diarios: lo que se usa al abrir el día d viene solo de días anteriores
    dO, dC = O[:, 0], C[np.arange(D), nbar - 1]
    dH, dL = np.nanmax(H, 1), np.nanmin(L, 1)
    prevC = np.r_[np.nan, dC[:-1]]
    trd = np.fmax(dH - dL, np.fmax(np.abs(dH - prevC), np.abs(dL - prevC)))
    atr_d = pd.Series(trd).rolling(14).mean().shift(1).to_numpy()
    sma20 = pd.Series(dC).rolling(20).mean().shift(1).to_numpy()
    tend = np.sign(prevC - sma20)
    v0 = V[:, 0]
    rvol = v0 / pd.Series(v0).rolling(14).mean().shift(1).to_numpy()
    mov = np.abs(C / dO[:, None] - 1)
    sig = pd.DataFrame(mov).rolling(14, min_periods=10).mean().shift(1).to_numpy()

    # Filtro de mercado: QQQ sobre/bajo su VWAP, alineado por fecha
    f_q, Q = a_matrices(datos[FILTRO])
    rellenar(Q)
    posq = {f: i for i, f in enumerate(f_q)}
    filas = np.array([posq.get(f, -1) for f in fechas])
    hay = filas >= 0
    Qa = {c: np.full(C.shape, np.nan) for c in Q}
    for c in Q:
        Qa[c][hay] = Q[c][filas[hay]]
    qvw, _ = vwap(Qa)
    with np.errstate(invalid="ignore"):
        qUp, qDn = Qa["close"] > qvw, Qa["close"] < qvw

    return SimpleNamespace(
        fechas=fechas, anios=np.array([f.year for f in fechas]), D=D, nbar=nbar,
        O=O, H=H, L=L, C=C, V=V, VW=VW, SD=SD, ATR5=ATR5, VMA=VMA, ema=ema,
        dO=dO, dC=dC, prevC=prevC, atr_d=atr_d, tend=tend, rvol=rvol, sig=sig, qUp=qUp, qDn=qDn,
    )


# ------------------------------------------------------------------ ESTRATEGIAS
# Cada estrategia marca señales al CIERRE de la vela k (se entra a la apertura de k+1) y
# define el stop (y el objetivo, si tiene) con información disponible en ese momento.

def _vacio(F):
    z, n = np.zeros(F.C.shape, bool), np.full(F.C.shape, np.nan)
    return z.copy(), z.copy(), n.copy(), n.copy()


def e_orb5_clasico(F, rr=10.0, m_stop=1.0):
    """Zarattini y Aziz (2023): el color de la 1ª vela de 5 min marca la dirección; entra a las
    9:35 con stop en el extremo de esa vela y objetivo 10R, o sale al cierre."""
    sigL, sigS, stopL, stopS = _vacio(F)
    c0, o0, h0, l0 = F.C[:, 0], F.O[:, 0], F.H[:, 0], F.L[:, 0]
    sigL[:, 0], sigS[:, 0] = c0 > o0, c0 < o0
    stopL[:, 0] = c0 - m_stop * (c0 - l0)
    stopS[:, 0] = c0 + m_stop * (h0 - c0)
    return dict(sigL=sigL, sigS=sigS, stopL=stopL, stopS=stopS, rr=rr, max_trades=1)


def e_orb5_rvol(F, rv_min=1.0, stop_atr=0.10):
    """Zarattini, Barbon y Aziz (2024): solo días con volumen de apertura alto; entra cuando una
    vela cierra fuera de la 1ª vela a favor de su color; stop = 10% del ATR diario; sale al cierre."""
    c0, o0 = F.C[:, 0], F.O[:, 0]
    ven = (KIDX >= 1) & (KIDX <= K_ULTIMA_SENAL)
    with np.errstate(invalid="ignore"):
        ok = (F.rvol >= rv_min)[:, None]
        sigL = ven & ok & (c0 > o0)[:, None] & (F.C > F.H[:, :1])
        sigS = ven & ok & (c0 < o0)[:, None] & (F.C < F.L[:, :1])
    stopL = F.C - stop_atr * F.atr_d[:, None]
    stopS = F.C + stop_atr * F.atr_d[:, None]
    return dict(sigL=sigL, sigS=sigS, stopL=stopL, stopS=stopS, rr=np.inf, max_trades=1)


def e_ruido(F, banda=1.0, freq=30):
    """Zarattini, Aziz y Barbon (2024): banda de 'ruido' = movimiento medio desde la apertura a esa
    hora en los últimos 14 días. Revisa cada `freq` min: entra si cierra fuera de la banda; sale si
    vuelve detrás de la banda/VWAP (stop móvil) o al cierre. Stop de emergencia a 2R."""
    arriba = np.fmax(F.dO, F.prevC)[:, None] * (1 + banda * F.sig)
    abajo = np.fmin(F.dO, F.prevC)[:, None] * (1 - banda * F.sig)
    chk = ((KIDX + 1) * 5) % freq == 0
    ven = chk & (KIDX <= K_ULTIMA_SENAL)
    nivelL, nivelS = np.fmax(arriba, F.VW), np.fmin(abajo, F.VW)
    with np.errstate(invalid="ignore"):
        sigL, sigS = ven & (F.C > arriba), ven & (F.C < abajo)
        exitL, exitS = chk & (F.C < nivelL), chk & (F.C > nivelS)
    return dict(sigL=sigL, sigS=sigS, stopL=nivelL, stopS=nivelS, exitL=exitL, exitS=exitS,
                rr=np.inf, stop_duro_R=2.0, max_trades=2)


def e_vwap_rev(F, banda=2.0, stop_sd=1.0):
    """Reversión a la media: desde las 10:00, si el precio se aleja `banda` desvíos del VWAP,
    apuesta a que vuelve al VWAP. Stop `stop_sd` desvíos más allá."""
    ven = (KIDX >= 5) & (KIDX <= K_ULTIMA_SENAL)
    with np.errstate(invalid="ignore"):
        sigL = ven & (F.C < F.VW - banda * F.SD)
        sigS = ven & (F.C > F.VW + banda * F.SD)
    stopL, stopS = F.C - stop_sd * F.SD, F.C + stop_sd * F.SD
    return dict(sigL=sigL, sigS=sigS, stopL=stopL, stopS=stopS, tgtL=F.VW, tgtS=F.VW, max_trades=2)


def e_ema(F, rapida=9, lenta=21, rr=2.0):
    """La más difundida en redes: cruce de EMAs a favor del VWAP. Stop en el mínimo/máximo de las
    últimas 5 velas; objetivo 2R."""
    ef, es = F.ema(rapida), F.ema(lenta)
    pf, ps = np.roll(ef, 1, 1), np.roll(es, 1, 1)
    ven = (KIDX >= 3) & (KIDX <= K_ULTIMA_SENAL)
    with np.errstate(invalid="ignore"):
        sigL = ven & (ef > es) & (pf <= ps) & (F.C > F.VW)
        sigS = ven & (ef < es) & (pf >= ps) & (F.C < F.VW)
    ll, hh = F.L.copy(), F.H.copy()
    for s in range(1, 5):
        ll[:, s:] = np.fmin(ll[:, s:], F.L[:, :-s])
        hh[:, s:] = np.fmax(hh[:, s:], F.H[:, :-s])
    return dict(sigL=sigL, sigS=sigS, stopL=ll, stopS=hh, rr=rr, max_trades=2)


def e_gap(F, gap_min=0.01, frac=1.0):
    """Cierre de gap: si abre con gap >= 1% y la 1ª vela va en contra del gap, apuesta a que el
    precio vuelve al cierre anterior. Stop en el extremo de la 1ª vela; si no llega, sale al cierre."""
    sigL, sigS, stopL, stopS = _vacio(F)
    tgtL, tgtS = np.full(F.C.shape, np.nan), np.full(F.C.shape, np.nan)
    c0, o0, h0, l0 = F.C[:, 0], F.O[:, 0], F.H[:, 0], F.L[:, 0]
    with np.errstate(invalid="ignore"):
        gap = F.dO / F.prevC - 1
        sigS[:, 0] = (gap >= gap_min) & (c0 < o0)
        sigL[:, 0] = (gap <= -gap_min) & (c0 > o0)
    stopS[:, 0], stopL[:, 0] = h0, l0
    tgtS[:, 0] = c0 - frac * (c0 - F.prevC)
    tgtL[:, 0] = c0 + frac * (F.prevC - c0)
    return dict(sigL=sigL, sigS=sigS, stopL=stopL, stopS=stopS, tgtL=tgtL, tgtS=tgtS, max_trades=1)


def e_orb_tend(F, or_min=15, rr=2.0, atr_mult=1.0):
    """El ORB anterior (rango de 15 min + VWAP + volumen + QQQ), pero solo a favor de la tendencia
    diaria (cierre de ayer vs media de 20 días) y 1 trade por día."""
    n = or_min // 5
    orh = np.nanmax(F.H[:, :n], 1)[:, None]
    orl = np.nanmin(F.L[:, :n], 1)[:, None]
    ven = (KIDX >= n) & (KIDX <= K_ULTIMA_SENAL)
    with np.errstate(invalid="ignore"):
        vol = F.V > F.VMA
        sigL = ven & (F.C > orh) & (F.C > F.VW) & vol & F.qUp & (F.tend > 0)[:, None]
        sigS = ven & (F.C < orl) & (F.C < F.VW) & vol & F.qDn & (F.tend < 0)[:, None]
    stopL = np.fmax(orl, F.C - atr_mult * F.ATR5)
    stopS = np.fmin(orh, F.C + atr_mult * F.ATR5)
    return dict(sigL=sigL, sigS=sigS, stopL=stopL, stopS=stopS, rr=rr, max_trades=1)


def grilla(**listas):
    return [dict(zip(listas, vals)) for vals in itertools.product(*listas.values())]


# (nombre, función, parámetros base fijados de antemano, grilla para medir robustez en TRAIN)
ESTRATEGIAS = [
    ("ORB 5 min clásico", e_orb5_clasico, dict(rr=10.0, m_stop=1.0),
     grilla(rr=[3.0, 5.0, 10.0], m_stop=[0.75, 1.0, 1.5])),
    ("ORB 5 min + volumen relativo", e_orb5_rvol, dict(rv_min=1.0, stop_atr=0.10),
     grilla(rv_min=[1.0, 1.5, 2.0], stop_atr=[0.05, 0.10, 0.20])),
    ("Momentum zona de ruido", e_ruido, dict(banda=1.0, freq=30),
     grilla(banda=[0.8, 1.0, 1.2], freq=[15, 30, 60])),
    ("Reversión al VWAP", e_vwap_rev, dict(banda=2.0, stop_sd=1.0),
     grilla(banda=[1.5, 2.0, 2.5], stop_sd=[0.5, 1.0, 1.5])),
    ("Cruce EMA 9/21 + VWAP", e_ema, dict(rapida=9, lenta=21, rr=2.0),
     [dict(rapida=a, lenta=b, rr=r) for a, b in ((5, 13), (9, 21), (12, 26)) for r in (1.5, 2.0, 3.0)]),
    ("Cierre de gap", e_gap, dict(gap_min=0.01, frac=1.0),
     grilla(gap_min=[0.005, 0.01, 0.015], frac=[0.5, 0.75, 1.0])),
    ("ORB 15 min + tendencia diaria", e_orb_tend, dict(or_min=15, rr=2.0),
     grilla(or_min=[5, 15, 30], rr=[1.5, 2.0, 3.0])),
]


# ------------------------------------------------------------------ SIMULACIÓN
def simular(F, E, dias):
    """Ejecuta las señales día por día con las reglas comunes de riesgo y ejecución.
    Dentro de una vela se asume lo peor: si toca stop y objetivo, cuenta el stop."""
    sigL, sigS, stopL, stopS = E["sigL"], E["sigS"], E["stopL"], E["stopS"]
    tgtL, tgtS = E.get("tgtL"), E.get("tgtS")
    exitL, exitS = E.get("exitL"), E.get("exitS")
    rr, duro, maxt = E.get("rr", np.inf), E.get("stop_duro_R"), E.get("max_trades", 2)
    hay = sigL | sigS
    equity, trades, curva = CAPITAL_INICIAL, [], []

    for d in dias:
        cand = np.flatnonzero(hay[d])
        if cand.size == 0:
            curva.append(equity)
            continue
        kfin = min(K_SALIDA, int(F.nbar[d]) - 1)
        o, h, l, c = F.O[d].tolist(), F.H[d].tolist(), F.L[d].tolist(), F.C[d].tolist()
        a5 = F.ATR5[d].tolist()
        eq0, nt, libre = equity, 0, 0
        for k in cand.tolist():
            if nt >= maxt or k >= kfin or k > K_ULTIMA_SENAL:
                break
            if k < libre:
                continue
            if equity <= eq0 * (1 - PERDIDA_DIARIA_MAX):
                break
            lado = 1 if sigL[d, k] else -1
            atr = a5[k]
            if not atr > 0:
                continue
            if lado * (o[k + 1] - c[k]) > ANTI_DESLIZ_ATR * atr:   # regla anti-deslizamiento
                continue
            entrada = o[k + 1] * (1 + lado * SLIPPAGE)
            dist = lado * (entrada - (stopL if lado == 1 else stopS)[d, k])
            if not dist > 0:          # stop inválido o ya superado
                continue
            dist = max(dist, MIN_STOP_ATR * atr)
            if tgtL is not None:
                tgt = (tgtL if lado == 1 else tgtS)[d, k]
                if not lado * (tgt - entrada) > 0:
                    continue
            elif np.isfinite(rr):
                tgt = entrada + lado * rr * dist
            else:
                tgt = None
            st = entrada - lado * (duro or 1.0) * dist
            salidas = None
            if exitL is not None:
                salidas = (exitL if lado == 1 else exitS)[d].tolist()
            qty = min(equity * RIESGO_POR_TRADE / dist, equity / entrada)

            j = k + 1
            while True:
                if lado == 1:
                    if l[j] <= st:
                        px, mot = min(st, o[j]), "SL"
                        break
                    if tgt is not None and h[j] >= tgt:
                        px, mot = max(tgt, o[j]), "TP"
                        break
                else:
                    if h[j] >= st:
                        px, mot = max(st, o[j]), "SL"
                        break
                    if tgt is not None and l[j] <= tgt:
                        px, mot = min(tgt, o[j]), "TP"
                        break
                if j >= kfin:
                    px, mot = c[j], "CIERRE"
                    break
                if salidas is not None and salidas[j]:
                    j += 1
                    px, mot = o[j], "SALIDA"
                    break
                j += 1

            px *= 1 - lado * SLIPPAGE
            pnl = lado * (px - entrada) * qty
            equity += pnl
            trades.append((F.fechas[d], int(F.anios[d]), "LONG" if lado == 1 else "SHORT",
                           round(entrada, 2), round(px, 2), mot, lado * (px - entrada) / dist, pnl))
            nt += 1
            libre = j + 1
        curva.append(equity)

    tr = pd.DataFrame(trades, columns=["fecha", "anio", "lado", "entrada", "salida", "motivo", "R", "pnl"])
    return tr, np.array(curva)


def metricas(tr, curva):
    if len(tr) == 0:
        return dict(trades=0, win=0.0, pf=0.0, expR=0.0, ret=0.0, dd=0.0, sharpe=0.0)
    g, p = tr.pnl[tr.pnl > 0].sum(), -tr.pnl[tr.pnl < 0].sum()
    eq = np.r_[CAPITAL_INICIAL, curva]
    dd = float((1 - eq / np.maximum.accumulate(eq)).max())
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    return dict(trades=len(tr), win=float((tr.pnl > 0).mean()), pf=float(g / p) if p > 0 else float("inf"),
                expR=float(tr.R.mean()), ret=float(eq[-1] / eq[0] - 1), dd=dd, sharpe=sharpe)


def evaluar(F, nombre, fn, base, grid, d_train, d_test):
    E = fn(F, **base)
    tr_is, c_is = simular(F, E, d_train)
    tr_oos, c_oos = simular(F, E, d_test)
    m_is, m_oos = metricas(tr_is, c_is), metricas(tr_oos, c_oos)

    filas = []
    for p in grid:
        t, c = simular(F, fn(F, **p), d_train)
        filas.append({**p, **metricas(t, c)})
    gr = pd.DataFrame(filas)
    pct = float((gr.expR > 0).mean())

    todos = pd.concat([tr_is.assign(set="train"), tr_oos.assign(set="test")], ignore_index=True)
    por_anio = todos.groupby("anio").R.agg(["count", "mean"]) if len(todos) else pd.DataFrame()
    anios_pos = int((por_anio["mean"] > 0).sum()) if len(por_anio) else 0
    ratio = m_oos["expR"] / m_is["expR"] if m_is["expR"] > 0 else 0.0
    comp = {
        f">= {MIN_TRADES} trades en total": m_is["trades"] + m_oos["trades"] >= MIN_TRADES,
        f"PF test > {MIN_PF}": m_oos["pf"] > MIN_PF,
        f"MaxDD test < {MAX_DD:.0%}": m_oos["dd"] < MAX_DD,
        f"Test >= {MIN_RATIO_OOS:.0%} del train": ratio >= MIN_RATIO_OOS,
        f">= {MIN_GRILLA:.0%} de la grilla positiva": pct >= MIN_GRILLA,
    }
    return dict(nombre=nombre, base=base, m_is=m_is, m_oos=m_oos, pct=pct, grilla=gr,
                por_anio=por_anio, anios_pos=anios_pos, n_anios=len(por_anio), comp=comp,
                n_ok=sum(comp.values()), pasa=all(comp.values()),
                trades=todos.assign(estrategia=nombre), doc=" ".join((fn.__doc__ or "").split()))


# ------------------------------------------------------------------ SALIDAS
def _pf(x):
    return "inf" if np.isinf(x) else f"{x:.2f}"


def fmt(m):
    return (f"{m['trades']} trades | Win {m['win']:.0%} | PF {_pf(m['pf'])} | {m['expR']:+.2f}R | "
            f"Ret {m['ret']:+.1%} | MaxDD {m['dd']:.1%} | Sharpe {m['sharpe']:.2f}")


def telegram(texto):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        return
    import requests
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": texto[:4000]}, timeout=20)
        print("Telegram:", r.status_code)
    except Exception as e:
        print("No se pudo enviar a Telegram:", e)


def anotar(titulo, texto):
    """Deja el texto como anotación de la ejecución de GitHub (se ve en la página del run)."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    t = texto.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::notice title={titulo}::{t}")


# ------------------------------------------------------------------ MAIN
def main():
    t0 = time.time()
    demo = "--demo" in sys.argv
    datos, feed = datos_demo() if demo else bajar_datos_alpaca()
    F = preparar(datos)
    print(f"Datos listos: {F.D} días ({time.time() - t0:.0f} s)")

    corte = int(F.D * TRAIN_FRAC)
    d_train, d_test = np.arange(corte), np.arange(corte, F.D)
    bh = F.dC[-1] / F.dO[corte] - 1

    res = []
    for nombre, fn, base, grid in ESTRATEGIAS:
        r = evaluar(F, nombre, fn, base, grid, d_train, d_test)
        res.append(r)
        print(f"{nombre}: {r['n_ok']}/5 ({time.time() - t0:.0f} s)")
    ranking = sorted(res, key=lambda r: (r["n_ok"], min(r["m_oos"]["pf"], 99)), reverse=True)
    ganadoras = [r["nombre"] for r in ranking if r["pasa"]]
    veredicto = (f"PASA: {', '.join(ganadoras)} -> candidata a paper trading" if ganadoras
                 else "ninguna pasa todas las compuertas -> no hay ventaja comprobada")
    periodo = f"{F.fechas[0]} a {F.fechas[-1]}"

    # Resumen para Telegram
    tg = [f"📊 {len(res)} estrategias de day trading - {SIMBOLO}",
          f"Datos {feed} 5 min | {periodo}",
          f"Train {len(d_train)} días / Test {len(d_test)} días (desde {F.fechas[corte]})",
          f"Buy&hold {SIMBOLO} en test: {bh:+.1%}",
          "", "Ranking por resultado en TEST:"]
    for i, r in enumerate(ranking, 1):
        m = r["m_oos"]
        tg.append(f"{i}) {'✅' if r['pasa'] else '❌'} {r['nombre']} - {r['n_ok']}/5 compuertas")
        tg.append(f"   {m['trades']} trades | PF {_pf(m['pf'])} | {m['expR']:+.2f}R | "
                  f"DD {m['dd']:.0%} | años positivos {r['anios_pos']}/{r['n_anios']}")
    tg += ["", f"Veredicto: {veredicto}"]
    resumen = "\n".join(tg)

    # Reporte completo
    rep = [f"# {len(res)} estrategias de day trading - {SIMBOLO} - {datetime.now():%Y-%m-%d}",
           f"Datos: {feed} | {periodo} | Train {len(d_train)} días / Test {len(d_test)} días",
           f"Buy & hold {SIMBOLO} en test: {bh:+.1%}", "",
           f"**Veredicto: {veredicto}**", ""]
    bloques = []
    for r in ranking:
        b = [r["doc"], f"Parámetros base: {r['base']}",
             f"- TRAIN: {fmt(r['m_is'])}", f"- TEST:  {fmt(r['m_oos'])}",
             *[f"- {'✅' if v else '❌'} {k}" for k, v in r["comp"].items()]]
        if r["n_anios"]:
            b.append("- Por año (R medio): " + " | ".join(
                f"{a}: {row['mean']:+.2f}R ({int(row['count'])})" for a, row in r["por_anio"].iterrows()))
        mejor = r["grilla"].sort_values("expR", ascending=False).iloc[0]
        params = ", ".join(f"{k}={float(mejor[k]):g}" for k in r["base"])
        b.append(f"- Grilla en TRAIN: {r['pct']:.0%} positiva; mejor {params} -> {mejor['expR']:+.2f}R (no se usa para el veredicto)")
        bloques.append((r["nombre"], "\n".join(b)))
        rep += [f"## {'✅' if r['pasa'] else '❌'} {r['nombre']} ({r['n_ok']}/5)", *b, ""]
    reporte = "\n".join(rep)

    with open("reporte_estrategias.md", "w", encoding="utf-8") as f:
        f.write(reporte)
    pd.concat([r["trades"] for r in res], ignore_index=True).to_csv("trades_estrategias.csv", index=False)
    print(reporte)
    print(f"\nTiempo total: {time.time() - t0:.0f} s")

    anotar("Resumen", resumen)
    for nombre, b in bloques:
        anotar(nombre, b)
    telegram(resumen)


if __name__ == "__main__":
    main()
