"""Download and cache Alpaca 1-minute bars (regular session only) for the research runs."""
import gzip
import os
import pickle
import time

import pandas as pd
import requests

DATA_URL = "https://data.alpaca.markets"
NY = "America/New_York"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _headers():
    return {
        "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
    }


def _get(session, url, params):
    for attempt in range(8):
        response = session.get(url, headers=_headers(), params=params, timeout=60)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(min(2 ** attempt, 60))
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()


def fetch_symbol(session, symbol, start, end, feed):
    params = {
        "timeframe": "1Min", "start": start, "end": end, "limit": 10000,
        "adjustment": "split", "feed": feed, "sort": "asc",
    }
    frames = []
    while True:
        data = _get(session, f"{DATA_URL}/v2/stocks/{symbol}/bars", params)
        bars = data.get("bars") or []
        if bars:
            page = pd.DataFrame(bars)[["t", "o", "h", "l", "c", "v"]]
            page["t"] = pd.to_datetime(page["t"], utc=True).dt.tz_convert(NY)
            page = page.set_index("t").between_time("09:30", "15:59")
            frames.append(page)
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        if len(frames) % 50 == 0 and frames:
            print(f"  {symbol}: {len(frames)} páginas, hasta {frames[-1].index[-1]:%Y-%m-%d}", flush=True)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype({"o": "float64", "h": "float64", "l": "float64", "c": "float64", "v": "float64"})


def load(symbols, start, end, feed="sip"):
    """Returns {symbol: DataFrame[o,h,l,c,v]} indexed by New York bar-start time."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"bars_1min_{feed}_{start[:10]}.pkl.gz")
    cached = {}
    if os.path.exists(path):
        with gzip.open(path, "rb") as file:
            cached = pickle.load(file)
    session = requests.Session()
    out, changed = {}, False
    for symbol in symbols:
        df = cached.get(symbol)
        if df is None:
            print(f"Descargando {symbol}...", flush=True)
            df = fetch_symbol(session, symbol, start, end, feed)
            changed = True
        elif df.index[-1] < pd.Timestamp(end).tz_convert(NY) - pd.Timedelta(days=4):
            print(f"Actualizando {symbol} desde {df.index[-1]:%Y-%m-%d}...", flush=True)
            newer = fetch_symbol(session, symbol, df.index[-1].tz_convert("UTC").isoformat(), end, feed)
            df = pd.concat([df, newer]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            changed = True
        out[symbol] = df
    if changed:
        cached.update(out)
        with gzip.open(path, "wb", compresslevel=3) as file:
            pickle.dump(cached, file)
    return out
