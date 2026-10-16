#!/usr/bin/env python3
"""Paired comparison of experiment conditions from experiments.py CSV output (standard library only).

For each scenario and outcome, every condition is compared with a reference condition on the same seeds:
mean paired difference, a 95% t interval, an exact sign-flip permutation p-value (Monte Carlo with a fixed
seed above 16 pairs), and a Holm-adjusted p-value across all tests in that scenario. A condition whose
outcome equals the reference on every seed is flagged: it cannot differ by construction, or it did not run.

    python analyze.py v4-latent-ablation.csv
    python analyze.py v3-ablation.csv --reference baseline --outcomes reward_per_agent_tick,alive
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import statistics

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
DEFAULT_OUTCOMES = ("reward_per_agent_tick", "alive", "mean_energy", "goals_reached", "deaths", "births")


def t975(df: int) -> float:
    for key in sorted(T975):
        if key >= df:
            return T975[key]
    return 1.96


def sign_flip_p(diffs: list[float], rng: random.Random) -> float:
    """Two-sided paired permutation test of 'no difference' (exact up to 16 pairs)."""
    observed = abs(sum(diffs))
    if all(d == 0 for d in diffs):
        return 1.0
    if len(diffs) <= 16:
        hits = total = 0
        for signs in itertools.product((1, -1), repeat=len(diffs)):
            total += 1
            hits += abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
        return hits / total
    draws = 20000
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(draws))
    return (hits + 1) / (draws + 1)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", help="one or more CSV files written by experiments.py")
    parser.add_argument("--reference", default=None, help="reference condition (default: v3_control, else baseline)")
    parser.add_argument("--outcomes", default=",".join(DEFAULT_OUTCOMES))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.csv:
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no rows")
    conditions = sorted({r["ablation"] for r in rows})
    reference = args.reference or ("v3_control" if "v3_control" in conditions
                                   else "baseline" if "baseline" in conditions else conditions[0])
    outcomes = [o for o in args.outcomes.split(",") if any(o in r and r[o] != "" for r in rows)]
    rng = random.Random(0)
    print(f"Reference condition: {reference}. Paired by seed. p-values: sign-flip test; Holm-adjusted within scenario.")
    for scenario in sorted({r["scenario"] for r in rows}):
        by: dict[str, dict[str, dict[str, str]]] = {}
        for r in rows:
            if r["scenario"] == scenario:
                by.setdefault(r["ablation"], {})[r["seed"]] = r
        print(f"\n## Scenario: {scenario}")
        if reference not in by:
            print("Reference condition missing for this scenario.")
            continue
        tests = []
        for condition in conditions:
            if condition == reference or condition not in by:
                continue
            seeds = sorted(set(by[condition]) & set(by[reference]))
            for outcome in outcomes:
                pairs = [(float(by[reference][s][outcome]), float(by[condition][s][outcome])) for s in seeds
                         if by[reference][s].get(outcome, "") != "" and by[condition][s].get(outcome, "") != ""]
                if not pairs:
                    continue
                diffs = [c - r for r, c in pairs]
                mean = statistics.fmean(diffs)
                if len(diffs) > 1:
                    half = t975(len(diffs) - 1) * statistics.stdev(diffs) / math.sqrt(len(diffs))
                    interval = f"[{mean - half:+.4g}, {mean + half:+.4g}]"
                else:
                    interval = "n/a (1 pair)"
                tests.append({"condition": condition, "outcome": outcome, "n": len(diffs),
                              "ref": statistics.fmean(r for r, _ in pairs), "cond": statistics.fmean(c for _, c in pairs),
                              "diff": mean, "ci": interval, "p": sign_flip_p(diffs, rng),
                              "identical": all(d == 0 for d in diffs)})
        for test, adjusted in zip(tests, holm([t["p"] for t in tests])):
            test["holm"] = adjusted
        print("| Condition | Outcome | Pairs | Reference mean | Condition mean | Difference | 95% CI | p | Holm p |")
        print("|---|---|---|---|---|---|---|---|---|")
        for t in tests:
            name = t["condition"] + (" (identical to reference on every seed)" if t["identical"] else "")
            print(f"| {name} | {t['outcome']} | {t['n']} | {t['ref']:.4g} | {t['cond']:.4g} | {t['diff']:+.4g} | "
                  f"{t['ci']} | {t['p']:.3g} | {t['holm']:.3g} |")


if __name__ == "__main__":
    main()
