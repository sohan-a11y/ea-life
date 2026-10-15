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
    parser.add_argument("--disable-world-model", action="store_true",
                        help="disable predictive transition learning")
    parser.add_argument("--disable-collective", action="store_true",
                        help="disable shared trail signals")
    parser.add_argument("--ablation-suite", action="store_true",
                        help="run matched baseline, predictive-only, collective-only and full conditions")
    parser.add_argument("--latent-ablation-suite", action="store_true",
                        help="compare no latent model, latent learning only, one-step and long-horizon imagination")
    parser.add_argument("--disable-latent-model", action="store_true")
    parser.add_argument("--disable-imagination", action="store_true")
    parser.add_argument("--imagination-horizon", type=int, default=4)
    parser.add_argument("--imagination-beam-width", type=int, default=3)
    parser.add_argument("--output", default="experiment-results.csv")
    args = parser.parse_args()
    if args.seeds < 1 or args.ticks < 1:
        parser.error("--seeds and --ticks must be positive")
    if args.imagination_horizon < 1 or args.imagination_horizon > 12:
        parser.error("--imagination-horizon must be between 1 and 12")
    if args.imagination_beam_width < 1 or args.imagination_beam_width > 8:
        parser.error("--imagination-beam-width must be between 1 and 8")
    if sum([args.ablation_suite,args.latent_ablation_suite]) > 1:
        parser.error("--ablation-suite and --latent-ablation-suite are mutually exclusive")
    if args.latent_ablation_suite and (args.disable_latent_model or args.disable_imagination):
        parser.error("--latent-ablation-suite cannot be combined with individual latent disable flags")
    if args.ablation_suite and (args.disable_world_model or args.disable_collective):
        parser.error("--ablation-suite cannot be combined with individual disable flags")
    return args

def run():
    args = parse_args()
    scenarios = SCENARIOS if args.all_scenarios else (args.scenario,)
    if args.latent_ablation_suite:
        latent_conditions = (("v3_control",False,False,1),
                             ("latent_model_only",True,False,1),
                             ("one_step_planning",True,True,1),
                             ("long_horizon_imagination",True,True,args.imagination_horizon))
        ablations = tuple((name,not args.disable_world_model,not args.disable_collective,
                           latent_model_enabled,imagination_enabled,horizon)
                          for name,latent_model_enabled,imagination_enabled,horizon in latent_conditions)
    elif args.ablation_suite:
        ablations = tuple((name,world_model_enabled,collective_enabled,
                           not args.disable_latent_model,not args.disable_imagination,
                           args.imagination_horizon)
                          for name,world_model_enabled,collective_enabled in
                          (("baseline",False,False),("predictive_only",True,False),
                           ("collective_only",False,True),("full",True,True)))
    else:
        ablations = (("custom",not args.disable_world_model,not args.disable_collective,
                      not args.disable_latent_model,not args.disable_imagination,
                      args.imagination_horizon),)
    rows = []
    for scenario in scenarios:
        for (ablation,world_model_enabled,collective_enabled,latent_model_enabled,
             imagination_enabled,imagination_horizon) in ablations:
            for offset in range(args.seeds):
                seed = args.seed_start + offset
                cfg = Config(grid_size=args.grid_size, initial_population=args.population,
                             max_population=args.max_population,
                             max_total_organisms=max(args.max_organisms, args.population),
                             scenario=scenario, evolution_enabled=not args.disable_evolution,
                             world_model_enabled=world_model_enabled,
                             collective_enabled=collective_enabled,
                             latent_model_enabled=latent_model_enabled,
                             imagination_enabled=imagination_enabled,
                             imagination_horizon=imagination_horizon,
                             imagination_beam_width=args.imagination_beam_width)
                sim = Simulation(config=cfg, seed=seed)
                for _ in range(args.ticks):
                    sim.step()
                metrics = sim.metrics()
                row = {"scenario":scenario,"ablation":ablation,"seed":seed,"ticks":sim.tick,
                       "evolution_enabled":cfg.evolution_enabled,
                       "world_model_enabled":world_model_enabled,
                       "collective_enabled":collective_enabled,
                       "latent_model_enabled":latent_model_enabled,
                       "imagination_enabled":imagination_enabled,
                       "imagination_horizon":imagination_horizon,
                       "imagination_beam_width":args.imagination_beam_width,**metrics}
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
