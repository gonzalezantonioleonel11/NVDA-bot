# Bot de trading NVDA (Alpaca PAPER + aprobación por Telegram)

Bot automático que analiza el gráfico de NVDA cada 5 minutos y, si encuentra una buena
operación, te manda un mensaje a Telegram con **botones de Aprobar / Rechazar**. Si
aprobás, envía a Alpaca una **orden bracket**: entrada a mercado con **stop loss** y
**take profit** ya puestos.

> ⚠️ Está fijado a la cuenta **PAPER** de Alpaca (`paper-api.alpaca.markets`). No usa dinero real.

## LONG y SHORT

El bot busca operaciones en **las dos direcciones**:

- 🟢 **LONG (compra)**: gana si el precio sube. Stop debajo de la entrada, take profit arriba.
- 🔴 **SHORT (venta en corto)**: gana si el precio baja. Stop arriba de la entrada, take profit abajo.

Podés limitarlo con la variable `TRADE_SIDES` (en *Settings → Secrets and variables →
Actions → Variables*): `BOTH` (por defecto), `LONG` o `SHORT`. Para los SHORT, la cuenta
de Alpaca tiene que permitir venta en corto (cuenta con margen y al menos $2.000), y la
acción tiene que estar disponible para pedirla prestada. El bot lo revisa antes de
proponer la operación.

## Cómo analiza el mercado

### 1. Tendencia: filtros obligatorios
1. ADX > 20 (hay tendencia, no es un mercado lateral)
2. EMA20 por encima o por debajo de la EMA50 (gráfico de 5 min)
3. Precio por encima o por debajo del VWAP del día
4. Tendencia de **1 hora** a favor (multi-timeframe)
5. Que el **mercado general** (SPY, QQQ y SMH) no esté todo en contra

### 2. Confirmaciones: hacen falta 6 de 11, y al menos un gatillo ⚡
- ⚡ Cruce de RSI (45 hacia arriba para LONG, 55 hacia abajo para SHORT)
- ⚡ Ruptura de estructura reciente (BOS / CHoCH)
- ⚡ Barrido de liquidez (sweep de un swing previo)
- ⚡ **Fibonacci**: retroceso a la zona dorada (0.5–0.786) del último impulso, con rebote
- ⚡ **Soporte o resistencia**: rebote en un soporte, rechazo en una resistencia o ruptura de un nivel clave
- Fair Value Gap (FVG) reciente
- Histograma MACD a favor
- Sesgo de estructura de mercado a favor
- Volumen por encima del promedio
- RSI sin sobrecompra (LONG) o sin sobreventa (SHORT)
- Mercado general a favor (al menos 2 de SPY, QQQ y SMH)

### 3. Niveles clave: soportes y resistencias
- Máximo, mínimo y cierre del día anterior
- Rango de apertura (primeros 15 min)
- Máximo y mínimo del día
- Zonas donde el precio rebotó 3 o más veces en los últimos 3 días
- Retrocesos de Fibonacci 0.382 / 0.5 / 0.618 / 0.786

### 4. Lo que pasa ahora, no solo el pasado
- **Mercado general en este momento**: tendencia actual de SPY (S&P 500), QQQ (Nasdaq) y SMH (semiconductores)
- **Noticias de NVDA de las últimas 24 h** (API de noticias de Alpaca), con alerta si hay
  noticias de los últimos 30 minutos o si se habla de resultados (earnings)
- **Precio en vivo** al momento de aprobar: si el mercado se movió demasiado, se cancela

> Ningún bot puede predecir el futuro. Lo que hace este es ver **dónde está parado el
> precio ahora**, frente a los niveles donde suele reaccionar y frente al resto del
> mercado, y darte esa información para que decidas vos con el botón.

## Stop loss y take profit inteligentes

- **Stop loss**: arranca en 1.5 × ATR. Si hay un soporte (o una resistencia, en un SHORT)
  justo ahí, el stop se corre **detrás del nivel**, para que no te saque un simple testeo.
  Nunca queda a más de 2.5 × ATR.
- **Take profit**: apunta a 2 veces el riesgo (R:R 1:2). Si hay una resistencia (o un
  soporte, en un SHORT) antes, el objetivo se pone **justo antes de ese nivel**.
- Si el nivel siguiente está tan cerca que el R:R queda por debajo de **1:1.5**, la
  operación **se descarta**: no se compra pegado a una resistencia ni se vende pegado a un soporte.

| Parámetro | Valor | Significado |
|---|---|---|
| `RISK_PCT` | 0.5 % | Se arriesga como máximo el 0.5 % del capital por operación |
| `ATR_MULT` | 1.5 | Stop base = 1.5 × ATR |
| `MAX_STOP_ATR` | 2.5 | Stop máximo cuando se corre detrás de un nivel |
| `RR` | 2.0 | Take profit buscado = 2 × el riesgo |
| `MIN_RR` | 1.5 | R:R mínimo para aceptar una operación |
| `MAX_NOTIONAL_PCT` | 20 % | Una posición no puede superar el 20 % del capital |
| `MAX_TRADES_DAY` | 3 | Máximo de operaciones por día |
| Horario | 9:45–15:20 NY | Ventana para abrir operaciones nuevas |
| Cierre forzado | 15:50 NY | Cierra todo antes del final de la sesión |

Después de que aprobás, el bot **vuelve a verificar** que el mercado siga abierto, que no
haya otra posición abierta, que el precio no se haya movido demasiado y que el R:R siga
siendo aceptable con el precio actual. Si no respondés en 3 minutos, la señal vence y no se opera.

## Ejemplo de mensaje

```
🟢 LONG — COMPRA NVDA
━━━━━━━━━━━━━━━━━━
💵 Entrada: $187.84 (a mercado)
🛑 Stop loss: $186.39 (riesgo $1.45/acción · debajo de Máx. apertura $186.52)
🎯 Take profit: $190.75 (ganancia $2.91/acción · 2R)
⚖️ Riesgo/beneficio: 1:2.0
📦 Cantidad: 37 acciones (~$6,950.20)
💸 Pérdida máx. si toca el stop: $53.77 (0.05% de la cuenta)
━━━━━━━━━━━━━━━━━━
📊 Confluencia 8/11 (mínimo 6)
✅ ... checklist completo ...
━━━━━━━━━━━━━━━━━━
🧱 Resistencias: ninguna cercana
🧱 Soportes: $187.81 Máximo del día · $186.52 Máx. apertura · $186.08 Máximo de ayer
📐 Fibonacci tramo alcista ...
🌎 Mercado ahora: SPY ▲ · QQQ ▲ · SMH ▬
📰 Noticias (24 h): ...
        [ ✅ APROBAR OPERACIÓN ]  [ ❌ RECHAZAR ]
```

## Configuración

1. **Alpaca**: creá una cuenta en https://alpaca.markets y generá las API keys de *Paper Trading*.
2. **Telegram**:
   - Hablá con `@BotFather`, creá un bot con `/newbot` y copiá el token.
   - Mandale cualquier mensaje a tu bot. Después abrí
     `https://api.telegram.org/bot<TOKEN>/getUpdates` y copiá `chat.id` (tu chat) y
     `from.id` (tu usuario).
3. En GitHub, andá a **Settings → Secrets and variables → Actions** y creá estos secrets:
   - `ALPACA_KEY`, `ALPACA_SECRET`
   - `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
   - `TELEGRAM_APPROVER_ID` (opcional pero recomendado: solo ese usuario puede aprobar)
4. El workflow `.github/workflows/bot.yml` corre cada 5 minutos en horario de mercado.
   GitHub solo ejecuta los `schedule` desde la **rama por defecto** (`main`), así que el
   código tiene que estar ahí.

## Probarlo

- **Diagnóstico**: en *Actions → nvda-bot-paper → Run workflow* se ejecuta en modo
  manual. Te manda a Telegram el saldo, el checklist del análisis actual, los
  soportes y resistencias, Fibonacci, el mercado general y las noticias. **Nunca envía órdenes.**
- **Tests locales**:
  ```bash
  pip install -r requirements.txt pytest
  pytest -q
  ```

## Avisos

- Los datos usan el feed gratuito **IEX** de Alpaca, que tiene menos volumen que el consolidado.
- Los `cron` de GitHub Actions pueden atrasarse unos minutos en horas de mucha carga.
- Esto es una herramienta educativa: probala bastante tiempo en paper antes de pensar en dinero real.
