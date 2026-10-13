# Reporte semanal de la cuenta paper de Alpaca -> Telegram
# Lee todas tus operaciones ejecutadas, arma cada trade (entrada + salida)
# y te manda las estadísticas por Telegram.

import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

ARG = timezone(timedelta(hours=-3))          # hora Argentina
BASE = "https://paper-api.alpaca.markets"    # cuenta PAPER
META = 50                                    # operaciones para sacar conclusiones


# ---------- Alpaca ----------

def alpaca_get(path, params):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def traer_ejecuciones():
    """Todas las ejecuciones (fills) de la cuenta, de la más vieja a la más nueva."""
    fills, token = [], None
    while True:
        params = {"direction": "asc", "page_size": 100}
        if token:
            params["page_token"] = token
        pagina = alpaca_get("/v2/account/activities/FILL", params)
        if not pagina:
            break
        fills.extend(pagina)
        if len(pagina) < 100:
            break
        token = pagina[-1]["id"]
    return fills


# ---------- Armar operaciones ----------

def leer_hora(texto):
    return datetime.fromisoformat(texto.replace("Z", "+00:00")).astimezone(ARG)


def armar_operaciones(fills):
    """Une compras y ventas en operaciones completas (de posición 0 a posición 0)."""
    posicion = defaultdict(float)
    abiertas, cerradas = {}, []

    for f in fills:
        simbolo = f["symbol"]
        precio = float(f["price"])
        restante = float(f["qty"])
        signo = 1 if f["side"].lower().startswith("buy") else -1
        hora = leer_hora(f["transaction_time"])

        while restante > 1e-9:
            actual = posicion[simbolo]
            if actual == 0:  # abre operación nueva
                abiertas[simbolo] = {
                    "simbolo": simbolo,
                    "lado": "long" if signo > 0 else "short",
                    "entrada": hora,
                    "flujo": 0.0,
                    "invertido": 0.0,
                }
                cantidad = restante
            elif (actual > 0) == (signo > 0):  # agranda la posición
                cantidad = restante
            else:  # achica o cierra
                cantidad = min(restante, abs(actual))

            op = abiertas[simbolo]
            op["flujo"] += -signo * cantidad * precio
            if (actual == 0) or ((actual > 0) == (signo > 0)):
                op["invertido"] += cantidad * precio
            posicion[simbolo] = actual + signo * cantidad
            restante -= cantidad

            if abs(posicion[simbolo]) < 1e-9:  # volvió a 0: operación cerrada
                posicion[simbolo] = 0.0
                op["salida"] = hora
                op["resultado"] = op["flujo"]
                cerradas.append(op)
                del abiertas[simbolo]

    return cerradas, list(abiertas.values())


# ---------- Estadísticas ----------

def plata(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def bloque_basico(ops):
    n = len(ops)
    ganadas = [o["resultado"] for o in ops if o["resultado"] > 0]
    perdidas = [o["resultado"] for o in ops if o["resultado"] <= 0]
    total = sum(o["resultado"] for o in ops)
    lineas = [
        f"Operaciones: {n} | Ganadas: {len(ganadas)} ({len(ganadas) / n:.0%})",
        f"Resultado: {plata(total)}",
    ]
    prom_g = sum(ganadas) / len(ganadas) if ganadas else 0
    prom_p = sum(perdidas) / len(perdidas) if perdidas else 0
    texto = f"Prom. ganancia: {plata(prom_g)} | Prom. pérdida: {plata(prom_p)}"
    if prom_g and prom_p:
        texto += f" (ratio {prom_g / abs(prom_p):.1f})"
    lineas.append(texto)
    lineas.append(f"Esperanza por operación: {plata(total / n)}")
    return lineas


def por_grupo(ops, clave):
    grupos = defaultdict(list)
    for o in ops:
        grupos[clave(o)].append(o)
    lineas = []
    for nombre in sorted(grupos):
        g = grupos[nombre]
        gan = sum(1 for o in g if o["resultado"] > 0)
        tot = sum(o["resultado"] for o in g)
        lineas.append(f"  {nombre}: {len(g)} ops, {gan / len(g):.0%} ganadas, {plata(tot)}")
    return lineas


def armar_mensaje(cerradas, abiertas, ahora):
    lunes = (ahora - timedelta(days=ahora.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0)
    semana = [o for o in cerradas if o["salida"] >= lunes]

    m = ["📊 Reporte semanal (cuenta paper)", ""]

    m.append(f"ESTA SEMANA (desde el lunes {lunes:%d/%m})")
    m += bloque_basico(semana) if semana else ["Sin operaciones cerradas."]
    m.append("")

    if not cerradas:
        m.append("Todavía no hay operaciones cerradas en la cuenta.")
    else:
        m.append("ACUMULADO (desde el inicio)")
        m += bloque_basico(cerradas)
        m.append(f"Progreso: {len(cerradas)}/{META} operaciones antes de sacar conclusiones")
        m.append("")
        m.append("Por lado:")
        m += por_grupo(cerradas, lambda o: o["lado"])
        m.append("Por hora de entrada (hora ARG):")
        m += por_grupo(cerradas, lambda o: f"{o['entrada']:%H} h")
        m.append("Por día:")
        dias = ["1 lun", "2 mar", "3 mié", "4 jue", "5 vie", "6 sáb", "7 dom"]
        m += por_grupo(cerradas, lambda o: dias[o["entrada"].weekday()])
        m.append("Por acción:")
        m += por_grupo(cerradas, lambda o: o["simbolo"])
        peor = min(cerradas, key=lambda o: o["resultado"])
        mejor = max(cerradas, key=lambda o: o["resultado"])
        m.append("")
        m.append(f"Peor: {peor['simbolo']} {peor['lado']} {peor['entrada']:%d/%m %H:%M} {plata(peor['resultado'])}")
        m.append(f"Mejor: {mejor['simbolo']} {mejor['lado']} {mejor['entrada']:%d/%m %H:%M} {plata(mejor['resultado'])}")

    if abiertas:
        m.append("")
        nombres = ", ".join(f"{o['simbolo']} {o['lado']}" for o in abiertas)
        m.append(f"⚠️ Posición abierta (no se cuenta): {nombres}")

    return "\n".join(m)


# ---------- Telegram ----------

def mandar_telegram(texto):
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/sendMessage"
    datos = urllib.parse.urlencode({
        "chat_id": os.environ["TELEGRAM_CHAT_ID"],
        "text": texto[:4000],
    }).encode()
    urllib.request.urlopen(urllib.request.Request(url, data=datos), timeout=30)


if __name__ == "__main__":
    try:
        cerradas, abiertas = armar_operaciones(traer_ejecuciones())
        mensaje = armar_mensaje(cerradas, abiertas, datetime.now(ARG))
    except Exception as e:
        mandar_telegram(f"⚠️ No pude armar el reporte semanal: {type(e).__name__}: {e}")
        raise
    mandar_telegram(mensaje)
    print(mensaje)
