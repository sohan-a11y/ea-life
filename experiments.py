#!/usr/bin/env python3
"""Run repeatable offline EA-Life experiments and export one row per seed/scenario."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from ea_life.engine import Config, SCENARIOS, Simulation

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20261015)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--max-population", type=int, default=16)
    parser.add_argument("--max-organisms", type=int, default=200)
    parser.add_argument("--disable-evolution", action="store_true")
    parser.add_argument("--output", default="experiment-results.csv")
    args = parser.parse_args()
    if args.seeds < 1 or args.ticks < 1:
        parser.error("--seeds and --ticks must be positive")
    return args

def run():
    args = parse_args()
    scenarios = SCENARIOS if args.all_scenarios else (args.scenario,)
    rows = []
    for scenario in scenarios:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            cfg = Config(grid_size=args.grid_size, initial_population=args.population,
                         max_population=args.max_population,
                         max_total_organisms=max(args.max_organisms, args.population),
                         scenario=scenario, evolution_enabled=not args.disable_evolution)
            sim = Simulation(config=cfg, seed=seed)
            for _ in range(args.ticks):
                sim.step()
            metrics = sim.metrics()
            row = {"scenario":scenario,"seed":seed,"ticks":sim.tick,
                   "evolution_enabled":cfg.evolution_enabled,**metrics}
            row["roles"] = json.dumps(metrics["roles"],sort_keys=True)
            rows.append(row)
            print(json.dumps(row,sort_keys=True),flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote %d experiment summaries to %s"%(len(rows),output),flush=True)

if __name__=="__main__":
    run()
