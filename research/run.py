"""
Investigación de estrategias intradía en acciones y ETFs de Alpaca, con validación walk-forward.

Para cada familia de estrategias y cada año de prueba, se eligen los parámetros mirando solo
los 3 años anteriores y se mide el año siguiente, que la elección no vio. El resultado
"fuera de muestra" es la unión de esos años de prueba.

Uso:  python -m research.run --start 2017-01-01 [--families ORB5,Ruido] [--symbols SPY,QQQ]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import data, engine, families  # noqa: E402

DEFAULT_SYMBOLS = "SPY,QQQ,NVDA,TSLA,AAPL,AMD,META,AMZN,MSFT,GOOGL,NFLX,AVGO"
BASE = engine.Costs()
STRESS = BASE.scaled(2.0)
RISKS = (0.005, 0.01, 0.02)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def walk_forward(variants, test_years, train_years=3, min_train=150):
    picks, oos = [], []
    for year in test_years:
        best = None
        for name, tr in variants.items():
            if len(tr) == 0:
                continue
            st = engine.trade_stats(tr[(tr["year"] >= year - train_years) & (tr["year"] < year)])
            if st["n"] < min_train or st["R"] <= 0:
                continue
            if best is None or st["t"] > best[1]["t"]:
                best = (name, st)
        picks.append((year, best[0] if best else None))
        if best:
            tr = variants[best[0]]
            oos.append(tr[tr["year"] == year])
    return picks, (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


def daily_t(trades, start, end, risk=0.01):
    """t-statistic of daily portfolio returns (same-day trades are correlated, so not per trade)."""
    daily, _ = engine.portfolio(trades, risk)
    if len(daily) == 0:
        return 0.0
    d = daily.reindex(pd.bdate_range(start, end), fill_value=0.0)
    return float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0


def checks(st, st_stress, weeks, t_daily):
    per_week = st["n"] / weeks if weeks else 0
    return {
        "PF fuera de muestra >= 1.15": st["pf"] >= 1.15,
        ">= 4 operaciones por semana": per_week >= 4,
        "gana en >= 60% de los años": st["yrs_pos"] >= 60,
        "sigue ganando con costos x2": st_stress["R"] > 0,
        f"significativo (t diario {t_daily:.2f} >= 2)": t_daily >= 2,
    }, per_week


def fmt_stats(st):
    return (f"{st['n']} ops, aciertos {st['wr']:.0f}%, PF {st['pf']:.2f}, "
            f"R promedio {st['exp']:+.3f}, R total {st['R']:+.1f}, t {st['t']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--families", default=",".join(families.FAMILIES))
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--train-years", type=int, default=3)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = data.load(symbols, f"{args.start}T00:00:00Z", end, args.feed)
    markets = {s: engine.Market(s, df) for s, df in bars.items() if len(df) > 1000}
    first_year = min(int(m.year[0]) for m in markets.values())
    last_year = max(int(m.year[-1]) for m in markets.values())
    test_years = list(range(first_year + args.train_years, last_year + 1))
    test_start = pd.Timestamp(f"{test_years[0]}-01-01")
    test_end = max(pd.Timestamp(m.days[-1]) for m in markets.values())
    weeks = (test_end - test_start).days / 7
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = [
        f"# Investigación intradía: {', '.join(markets)}",
        "",
        f"Datos `{args.feed}` desde {args.start}: señales con velas de 5 minutos, ejecución simulada minuto a minuto. Años fuera de muestra: "
        f"{test_years[0]}–{test_years[-1]} (cada año elegido solo con los {args.train_years} anteriores).",
        f"Costos base: entrada {BASE.entry_bps} pb, stop {BASE.stop_bps} pb, salida {BASE.exit_bps} pb; "
        "estrés: el doble.",
        "",
    ]
    summary_rows, all_variants, final = [], [], []
    for fam in [f.strip() for f in args.families.split(",")]:
        print(f"Simulando familia {fam}...", flush=True)
        gen = families.FAMILIES[fam]
        base = dict(gen(markets, BASE))
        stress = dict(gen(markets, STRESS))
        for name, tr in base.items():
            st = engine.trade_stats(tr)
            all_variants.append({"familia": fam, "variante": name, **{k: st[k] for k in ("n", "wr", "pf", "exp", "R", "t", "yrs_pos")}})
        picks, oos = walk_forward(base, test_years, args.train_years)
        oos_stress = pd.concat([stress[n][stress[n]["year"] == y] for y, n in picks if n], ignore_index=True) \
            if any(n for _, n in picks) else pd.DataFrame()
        st, st_s = engine.trade_stats(oos), engine.trade_stats(oos_stress)
        t_d = daily_t(oos, test_start, test_end) if len(oos) else 0.0
        ok, per_week = checks(st, st_s, weeks, t_d)
        passed = all(ok.values())
        summary_rows.append((fam, st, st_s, per_week, passed))
        if len(oos):
            oos.to_csv(os.path.join(OUT_DIR, f"oos_{fam.replace('/', '-')}.csv"), index=False)

        final.append(f"{fam}: {'APRUEBA' if passed else 'no aprueba'} | {fmt_stats(st)} | {per_week:.1f} ops/sem | "
                     f"x2 costos R {st_s['R']:+.1f} | t diario {t_d:.2f} | "
                     + " | ".join(f"{k}: {'si' if v else 'NO'}" for k, v in ok.items()))
        lines += [f"## {fam} {'✅ APRUEBA' if passed else '❌ no aprueba'}", "",
                  f"- Fuera de muestra: {fmt_stats(st)}; {per_week:.1f} ops/semana",
                  f"- Con costos x2: {fmt_stats(st_s)}"]
        lines += [f"- {k}: {'sí' if v else 'NO'}" for k, v in ok.items()]
        lines += ["", "| Año | Variante elegida (con años anteriores) | Ops | R del año |", "|---|---|---|---|"]
        for year, name in picks:
            yr = oos[oos["year"] == year] if len(oos) else oos
            lines.append(f"| {year} | {name or 'ninguna calificó'} | {len(yr)} | {yr['r'].sum() if len(yr) else 0:+.1f} |")
        if len(oos):
            lines += ["", "| Riesgo por op. | Mes promedio | Mes mediano | Meses >= 3% | Meses negativos | Peor mes | Anual (CAGR) | Máx. caída |",
                      "|---|---|---|---|---|---|---|---|"]
            for risk in RISKS:
                daily, _ = engine.portfolio(oos, risk)
                ms = engine.monthly_stats(daily, test_start, test_end)
                lines.append(
                    f"| {risk * 100:.1f}% | {ms['mean_m']:+.2f}% | {ms['median_m']:+.2f}% | {ms['pct_ge3']:.0f}% | "
                    f"{ms['pct_neg']:.0f}% | {ms['worst_m']:+.1f}% | {ms['cagr']:+.1f}% | {ms['max_dd']:.1f}% |")
                final.append(f"   riesgo {risk * 100:.1f}%: mes prom {ms['mean_m']:+.2f}% mediano {ms['median_m']:+.2f}% "
                             f">=3% {ms['pct_ge3']:.0f}% neg {ms['pct_neg']:.0f}% peor {ms['worst_m']:+.1f}% "
                             f"CAGR {ms['cagr']:+.1f}% maxDD {ms['max_dd']:.1f}%")
            for year, name in picks:
                yr = oos[oos["year"] == year]
                final.append(f"   {year}: {name or '-'} -> {len(yr)} ops, R {yr['r'].sum() if len(yr) else 0:+.1f}")
        lines.append("")

    pd.DataFrame(all_variants).to_csv(os.path.join(OUT_DIR, "variantes.csv"), index=False)
    lines += ["## Resumen", "", "| Familia | Ops fuera de muestra | Ops/semana | PF | R promedio | PF con costos x2 | Aprueba |",
              "|---|---|---|---|---|---|---|"]
    for fam, st, st_s, per_week, passed in summary_rows:
        lines.append(f"| {fam} | {st['n']} | {per_week:.1f} | {st['pf']:.2f} | {st['exp']:+.3f} | {st_s['pf']:.2f} | "
                     f"{'✅' if passed else '❌'} |")
    lines += ["", "## Todas las variantes (período completo, dentro de muestra: solo de referencia)", "",
              "| Familia | Variante | Ops | Aciertos | PF | R prom. | t |", "|---|---|---|---|---|---|---|"]
    for v in sorted(all_variants, key=lambda v: -v["t"]):
        lines.append(f"| {v['familia']} | {v['variante']} | {v['n']} | {v['wr']:.0f}% | {v['pf']:.2f} | {v['exp']:+.3f} | {v['t']:.2f} |")

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "informe.md"), "w", encoding="utf-8") as file:
        file.write(report + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as file:
            file.write(report + "\n")
    print(report)
    print("\n===== RESUMEN FINAL =====")
    print("\n".join(final))
    top = sorted(all_variants, key=lambda v: -v["t"])[:15]
    print("Mejores variantes dentro de muestra (referencia):")
    for v in top:
        print(f"   {v['variante']}: {v['n']} ops PF {v['pf']:.2f} R prom {v['exp']:+.3f} t {v['t']:.2f}")


if __name__ == "__main__":
    main()
