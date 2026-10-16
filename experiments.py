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
    parser.add_argument("--imagination-bonus", type=float, default=0.45,
                        help="score bonus for the imagined plan's first action (0 = plan has no influence)")
    parser.add_argument("--heldout-ticks", type=int, default=0,
                        help="freeze the latent model for the last N ticks and report held-out prediction error")
    parser.add_argument("--output", default="experiment-results.csv")
    args = parser.parse_args()
    if args.seeds < 1 or args.ticks < 1:
        parser.error("--seeds and --ticks must be positive")
    if args.imagination_horizon < 1 or args.imagination_horizon > 12:
        parser.error("--imagination-horizon must be between 1 and 12")
    if args.imagination_beam_width < 1 or args.imagination_beam_width > 8:
        parser.error("--imagination-beam-width must be between 1 and 8")
    if not 0.0 <= args.imagination_bonus <= 10.0:
        parser.error("--imagination-bonus must be between 0 and 10")
    if not 0 <= args.heldout_ticks < args.ticks:
        parser.error("--heldout-ticks must be at least 0 and smaller than --ticks")
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
        # Preserve the v3 protocol: neural latent learning and imagination are off
        # in all four conditions, so this suite isolates the original v3 pathways.
        ablations = tuple((name,world_model_enabled,collective_enabled,False,False,1)
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
                             imagination_beam_width=args.imagination_beam_width,
                             imagination_bonus=args.imagination_bonus)
                sim = Simulation(config=cfg, seed=seed)
                for t in range(args.ticks):
                    if args.heldout_ticks and t == args.ticks - args.heldout_ticks:
                        sim.freeze_latent_model()
                    sim.step()
                metrics = sim.metrics()
                if args.heldout_ticks and latent_model_enabled:
                    metrics.update(sim.latent_heldout_metrics())
                row = {"scenario":scenario,"ablation":ablation,"seed":seed,"ticks":sim.tick,
                       "evolution_enabled":cfg.evolution_enabled,
                       "world_model_enabled":world_model_enabled,
                       "collective_enabled":collective_enabled,
                       "latent_model_enabled":latent_model_enabled,
                       "imagination_enabled":imagination_enabled,
                       "imagination_horizon":imagination_horizon,
                       "imagination_beam_width":args.imagination_beam_width,
                       "imagination_bonus":args.imagination_bonus,
                       "heldout_ticks":args.heldout_ticks,**metrics}
                row["roles"] = json.dumps(metrics["roles"],sort_keys=True)
                rows.append(row)
                print(json.dumps(row,sort_keys=True),flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as handle:
        fieldnames=[]
        for row in rows:                     # rows differ: held-out columns exist only with the latent model
            fieldnames.extend(key for key in row if key not in fieldnames)
        writer=csv.DictWriter(handle,fieldnames=fieldnames,restval="")
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote %d experiment summaries to %s"%(len(rows),output),flush=True)

if __name__=="__main__":
    run()
