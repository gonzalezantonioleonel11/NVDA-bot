"""Sector (industry group) of each symbol in the ORB universe, used to avoid stacking correlated
trades: stocks of the same group tend to break out together on the same news."""

GROUPS = {
    "índices": ["SPY", "QQQ", "IWM"],
    "semiconductores": ["NVDA", "AVGO", "AMD", "INTC", "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ARM"],
    "software": ["MSFT", "ORCL", "CRM", "ADBE", "PLTR", "SNOW", "CRWD", "PANW", "NOW", "SHOP"],
    "hardware": ["AAPL", "DELL", "ANET", "SMCI"],
    "internet y medios": ["GOOGL", "META", "NFLX", "PINS", "SNAP", "RBLX", "ROKU", "DIS"],
    "telecomunicaciones": ["T", "VZ", "TMUS", "CMCSA"],
    "autos": ["TSLA", "GM", "F", "RIVN", "LCID", "NIO"],
    "china": ["BABA", "PDD", "JD"],
    "comercio y consumo": ["AMZN", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD"],
    "viajes y plataformas": ["UBER", "ABNB", "DKNG"],
    "consumo básico": ["WMT", "COST", "KO", "PEP", "PG"],
    "bancos": ["JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW"],
    "pagos y fintech": ["V", "MA", "PYPL", "AXP", "SOFI", "HOOD", "AFRM", "UPST"],
    "cripto": ["COIN", "MSTR", "MARA", "RIOT"],
    "energía": ["XOM", "CVX", "OXY", "COP", "SLB"],
    "industria y defensa": ["BA", "CAT", "DE", "GE", "LMT", "RTX"],
    "salud": ["UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "MRNA", "BMY", "GILD", "AMGN"],
}

SECTOR = {symbol: group for group, symbols in GROUPS.items() for symbol in symbols}


def sector(symbol):
    return SECTOR.get(symbol, symbol)  # unknown symbols count as their own group
