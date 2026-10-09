# Bot de trading NVDA (Alpaca PAPER + aprobación por Telegram)

Bot automático que analiza el gráfico de NVDA cada 5 minutos y, si encuentra una buena
operación, te manda un mensaje a Telegram con **botones de Aprobar / Rechazar**. Si
aprobás, envía a Alpaca una **orden bracket**: entrada a mercado con **stop loss** y
**take profit** ya puestos.

> ⚠️ Está fijado a la cuenta **PAPER** de Alpaca (`paper-api.alpaca.markets`). No usa dinero real.

## Cómo analiza el gráfico (velas de 5 min)

**Filtros obligatorios** (todos tienen que cumplirse):
1. ADX > 20: hay tendencia
2. EMA20 por encima o por debajo de la EMA50
3. Precio por encima o por debajo del VWAP del día
4. Tendencia de 1 hora a favor (multi-timeframe: cierre vs EMA20 horaria con pendiente)

**Confirmaciones** (se necesitan al menos 5 de 8, y una tiene que ser un gatillo ⚡):
- ⚡ Cruce de RSI (45 hacia arriba para long, 55 hacia abajo para short)
- ⚡ Ruptura de estructura reciente (BOS / CHoCH)
- ⚡ Barrido de liquidez (sweep de un swing previo)
- Fair Value Gap (FVG) reciente
- Histograma MACD a favor
- Sesgo de estructura de mercado a favor
- Volumen por encima del promedio de 20 velas
- RSI sin sobrecompra (long) o sin sobreventa (short)

El mensaje de Telegram muestra el checklist completo y la puntuación, por ejemplo `6/8`.

## Gestión del riesgo (automática)

| Parámetro | Valor | Significado |
|---|---|---|
| `RISK_PCT` | 0.5 % | Se arriesga como máximo el 0.5 % del capital por operación |
| `ATR_MULT` | 1.5 | Stop loss = 1.5 × ATR desde la entrada |
| `RR` | 2.0 | Take profit = 2 × la distancia del stop (R:R 1:2) |
| `MAX_NOTIONAL_PCT` | 20 % | Una posición no puede superar el 20 % del capital |
| `MAX_TRADES_DAY` | 3 | Máximo de operaciones por día |
| Horario | 9:45–15:20 NY | Ventana para abrir operaciones nuevas |
| Cierre forzado | 15:50 NY | Cierra todo antes del final de la sesión |

Después de que aprobás, el bot **vuelve a verificar** que el mercado siga abierto, que no
haya otra posición abierta y que el precio no se haya movido demasiado. Recién ahí
recalcula el SL y el TP sobre el precio actual y envía la orden. Si no respondés en
3 minutos, la señal vence y no se opera.

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
  manual. Te manda a Telegram el saldo, el estado del mercado y el checklist del análisis
  actual. **Nunca envía órdenes.**
- **Tests locales**:
  ```bash
  pip install -r requirements.txt pytest
  pytest -q
  ```

## Avisos

- Los datos usan el feed gratuito **IEX** de Alpaca, que tiene menos volumen que el consolidado.
- Los `cron` de GitHub Actions pueden atrasarse unos minutos en horas de mucha carga.
- Esto es una herramienta educativa: probala bastante tiempo en paper antes de pensar en dinero real.
