"""
Investigación de estrategias intradía en acciones y ETFs de Alpaca, con validación walk-forward.

Usa los últimos 2 años. Para cada familia y cada trimestre de prueba, se eligen los parámetros
mirando solo los 6 meses anteriores y se mide el trimestre siguiente, que la elección no vio.
El resultado "fuera de muestra" es la unión de esos trimestres.

Uso:  python -m research.run [--years 2] [--families ORB5,Ruido] [--symbols SPY,QQQ]
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


def month_index(tr):
    days = pd.to_datetime(pd.Series(tr["day"]))
    return (days.dt.year * 12 + days.dt.month - 1).to_numpy()


def label(mi):
    return f"{mi // 12}-{mi % 12 + 1:02d}"


def walk_forward(variants, windows, min_train=40):
    """windows: (train_start, test_start, test_end) month indices. Picks by t-stat on the train months."""
    months = {name: month_index(tr) if len(tr) else np.array([]) for name, tr in variants.items()}
    picks, oos = [], []
    for train_start, test_start, test_end in windows:
        best = None
        for name, tr in variants.items():
            if len(tr) == 0:
                continue
            mi = months[name]
            st = engine.trade_stats(tr[(mi >= train_start) & (mi < test_start)])
            if st["n"] < min_train or st["R"] <= 0:
                continue
            if best is None or st["t"] > best[1]["t"]:
                best = (name, st)
        window = f"{label(test_start)} a {label(test_end - 1)}"
        if best:
            tr, mi = variants[best[0]], months[best[0]]
            part = tr[(mi >= test_start) & (mi < test_end)]
            oos.append(part)
            picks.append((window, best[0], len(part), float(part["r"].sum()) if len(part) else 0.0))
        else:
            picks.append((window, None, 0, 0.0))
    return picks, (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


def daily_t(trades, start, end, risk=0.01):
    """t-statistic of daily portfolio returns (same-day trades are correlated, so not per trade)."""
    daily, _ = engine.portfolio(trades, risk)
    if len(daily) == 0:
        return 0.0
    d = daily.reindex(pd.bdate_range(start, end), fill_value=0.0)
    return float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0


def weeks_total(tr):
    days = pd.to_datetime(pd.Series(tr["day"]))
    return max((days.max() - days.min()).days / 7, 1)


def compare_to_base(variants, windows, weeks, final):
    """Each filter against the unfiltered 'base' variant: hit rate, R per trade, and in how many
    quarters (out of sample windows) its total R beat the base."""
    base = variants["base"]
    base_mi = month_index(base)
    rows = ["", "| Filtro | Ops/sem | Aciertos | R prom. | R total | PF | t | Trimestres mejor que base |",
            "|---|---|---|---|---|---|---|---|"]
    final.append("   Comparación de filtros (período completo; trimestres = cuántos de los fuera de muestra supera a la base):")
    for name, tr in variants.items():
        st = engine.trade_stats(tr)
        better = 0
        if len(tr):
            mi = month_index(tr)
            for _, test_start, test_end in windows:
                r_var = tr["r"][(mi >= test_start) & (mi < test_end)].sum()
                r_base = base["r"][(base_mi >= test_start) & (base_mi < test_end)].sum()
                better += r_var > r_base
        line = (f"{st['n'] / weeks:.1f} | {st['wr']:.0f}% | {st['exp']:+.3f} | {st['R']:+.1f} | {st['pf']:.2f} | "
                f"{st['t']:.2f} | {better if name != 'base' else '-'}/{len(windows)}")
        rows.append(f"| {name} | {line} |")
        final.append(f"     {name:24s} {line.replace(' | ', '  ')}")
    return rows


def checks(st, st_stress, weeks, t_daily, picks):
    per_week = st["n"] / weeks if weeks else 0
    pos_windows = sum(1 for p in picks if p[3] > 0) / len(picks) * 100 if picks else 0
    return {
        "PF fuera de muestra >= 1.15": st["pf"] >= 1.15,
        ">= 4 operaciones por semana": per_week >= 4,
        f"gana en >= 60% de los trimestres ({pos_windows:.0f}%)": pos_windows >= 60,
        "sigue ganando con costos x2": st_stress["R"] > 0,
        f"significativo (t diario {t_daily:.2f} >= 2)": t_daily >= 2,
    }, per_week


def fmt_stats(st):
    return (f"{st['n']} ops, aciertos {st['wr']:.0f}%, PF {st['pf']:.2f}, "
            f"R promedio {st['exp']:+.3f}, R total {st['R']:+.1f}, t {st['t']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=2.0, help="años de datos (por defecto 2)")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--families", default=",".join(families.FAMILIES))
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--train-months", type=int, default=6)
    ap.add_argument("--test-months", type=int, default=3)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    now = pd.Timestamp.now(tz="UTC")
    end = (now - pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (now - pd.DateOffset(months=round(args.years * 12))).strftime("%Y-%m-01")
    bars = data.load(symbols, f"{start}T00:00:00Z", end, args.feed)
    markets = {s: engine.Market(s, df) for s, df in bars.items() if len(df) > 1000}
    first = min(pd.Timestamp(m.days[0]) for m in markets.values())
    last = max(pd.Timestamp(m.days[-1]) for m in markets.values())
    first_mi, last_mi = first.year * 12 + first.month - 1, last.year * 12 + last.month - 1
    windows = []
    test_mi = first_mi + args.train_months
    while test_mi <= last_mi:
        windows.append((test_mi - args.train_months, test_mi, min(test_mi + args.test_months, last_mi + 1)))
        test_mi += args.test_months
    test_start = pd.Timestamp(year=windows[0][1] // 12, month=windows[0][1] % 12 + 1, day=1)
    test_end = last
    weeks = (test_end - test_start).days / 7
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = [
        f"# Investigación intradía: {', '.join(markets)}",
        "",
        f"Datos `{args.feed}` desde {first:%Y-%m-%d} hasta {last:%Y-%m-%d}: señales con velas de 5 minutos, "
        f"ejecución simulada minuto a minuto. Fuera de muestra desde {test_start:%Y-%m}: cada trimestre usa "
        f"los parámetros elegidos con los {args.train_months} meses anteriores.",
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
        picks, oos = walk_forward(base, windows)
        stress_parts = []
        for (train_start, test_start_mi, test_end_mi), (_, name, _, _) in zip(windows, picks):
            if name and len(stress[name]):
                mi = month_index(stress[name])
                stress_parts.append(stress[name][(mi >= test_start_mi) & (mi < test_end_mi)])
        oos_stress = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
        st, st_s = engine.trade_stats(oos), engine.trade_stats(oos_stress)
        t_d = daily_t(oos, test_start, test_end) if len(oos) else 0.0
        ok, per_week = checks(st, st_s, weeks, t_d, picks)
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
        lines += ["", "| Trimestre | Variante elegida (con los meses anteriores) | Ops | R |", "|---|---|---|---|"]
        for window, name, n, r in picks:
            lines.append(f"| {window} | {name or 'ninguna calificó'} | {n} | {r:+.1f} |")
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
            for window, name, n, r in picks:
                final.append(f"   {window}: {name or '-'} -> {n} ops, R {r:+.1f}")
        if "base" in base:
            lines += compare_to_base(base, windows, weeks_total(base["base"]), final)
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
