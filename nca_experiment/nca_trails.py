"""Pre-registered NCA trail experiment (16 Oct 2026): does a learned local trail rule beat the hand-coded trails?

Run from the repository root (numpy required):
    python nca_experiment/nca_trails.py sanity nca_experiment/results/nca_sanity.json
    python nca_experiment/nca_trails.py screen nca_experiment/results/nca_screen.json 6
    python nca_experiment/nca_trails.py es     nca_experiment/results/nca_es_linear.json linear 5
    python nca_experiment/nca_trails.py es     nca_experiment/results/nca_es_nca.json nca 5
    python nca_experiment/nca_trails.py test   nca_experiment/results/nca_test.json nca_experiment/results/nca_selected.json 6
Set OMP_NUM_THREADS=1 (and OPENBLAS_NUM_THREADS=1) when running several workers, or they compete for cores.
The protocol, seeds and success rule are fixed in nca_experiment/nca_trails_prereg.json.
"""
import hashlib, itertools, json, math, os, sys, time
import numpy as np

KEYS = ("reward_per_agent_tick", "mean_energy", "alive", "deaths", "births", "resource_coverage", "goals_reached",
        "resource_trail_mass", "danger_trail_mass", "collective_signal_coverage")
_LOADED = {}

def _engine(root):
    if root not in _LOADED:
        for m in [m for m in sys.modules if m == "ea_life" or m.startswith("ea_life.")]: del sys.modules[m]
        sys.path.insert(0, root)
        import ea_life.engine as E
        tr = None
        try:
            import ea_life.trail_rules as tr
        except ImportError:
            pass
        sys.path.remove(root)
        _LOADED.clear(); _LOADED[root] = (E, tr)
    return _LOADED[root]

def run_job(job):
    root, spec, seed, scenario, ticks = job
    E, tr = _engine(root)
    cfg = E.Config(grid_size=48, initial_population=8, max_population=16, max_total_organisms=200, scenario=scenario,
                   latent_model_enabled=False, imagination_enabled=False, collective_enabled=(spec["kind"] != "no_trails"))
    sim = E.Simulation(cfg, seed=seed)
    if tr is not None and spec["kind"] not in ("engine", "no_trails"):
        sim.trail_rule = tr.make_rule(spec)
    t0 = time.time()
    for _ in range(ticks): sim.step()
    m = sim.metrics()
    h = hashlib.sha256(json.dumps([sim.resource_trails, sim.danger_trails, sim.resources,
                                   [(a["x"], a["y"], round(a["energy"], 9)) for a in sim.agents]]).encode()).hexdigest()[:16]
    return {**{k: m[k] for k in KEYS}, "state_hash": h, "seconds": round(time.time() - t0, 2)}

def pool_map(jobs, workers=6, tag="batch"):
    """Run jobs in separate worker processes (file in, file out); results come back in job order."""
    import subprocess, tempfile
    chunks = [list(range(i, len(jobs), workers)) for i in range(min(workers, len(jobs)))]
    tmp = tempfile.mkdtemp(prefix="nca_")
    procs = []
    for w, idx in enumerate(chunks):
        fin, fout = os.path.join(tmp, f"{tag}_{w}_in.json"), os.path.join(tmp, f"{tag}_{w}_out.json")
        json.dump([jobs[i] for i in idx], open(fin, "w"))
        procs.append((idx, fout, subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", fout, fin])))
    res = [None] * len(jobs)
    for idx, fout, p in procs:
        if p.wait() != 0: raise RuntimeError("worker failed: " + fout)
        for i, r in zip(idx, json.load(open(fout))): res[i] = r
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return res

def signflip_p(d):
    d = np.asarray(d, float); n = len(d); obs = abs(d.sum())
    if n <= 20:
        hits = 0; tot = 0
        for chunk in range(0, 2 ** n, 1 << 16):
            idx = np.arange(chunk, min(2 ** n, chunk + (1 << 16)), dtype=np.int64)
            signs = 1 - 2 * ((idx[:, None] >> np.arange(n)) & 1)
            s = np.abs(signs @ d); hits += int((s >= obs - 1e-12).sum()); tot += len(idx)
        return hits / tot
    rng = np.random.default_rng(0); signs = rng.choice([-1, 1], size=(200000, n)); return float((np.abs(signs @ d) >= obs - 1e-12).mean())

if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "worker":
        json.dump([run_job(tuple(j)) for j in json.load(open(sys.argv[3]))], open(out, "w")); sys.exit(0)
    V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, V5); import ea_life.trail_rules as TR; sys.path.remove(V5)
    for m in [m for m in sys.modules if m == "ea_life" or m.startswith("ea_life.")]: del sys.modules[m]
    P = json.load(open(os.path.join(HERE, "nca_trails_prereg.json")))
    PROTO_TICKS, TRAIN, VAL, TEST = P["protocol"]["ticks"], P["seeds"]["train"], P["seeds"]["validation"], P["seeds"]["test"]
    FIXED = {f"ca_p{'cur' if p == (0.985, 0.96) else 'slow'}_k{k}": {"kind": "fixed_ca", "persistence": list(p), "kappa": k}
             for p in ((0.985, 0.96), (0.995, 0.985)) for k in (0.0, 0.1, 0.25) if not (p == (0.985, 0.96) and k == 0.0)}
    if mode == "screen":
        arms = {"reference": {"kind": "engine"}, "no_trails": {"kind": "no_trails"}, **FIXED}
        jobs = [(V5, spec, s, "forage", PROTO_TICKS) for spec in arms.values() for s in VAL]
        res = pool_map(jobs, workers=int(sys.argv[3]) if len(sys.argv) > 3 else 6, tag="screen")
        outd = {name: {"spec": spec, "by_seed": {}} for name, spec in arms.items()}
        for (root, spec, s, sc, t), r in zip(jobs, res):
            name = next(k for k, v in arms.items() if v is spec); outd[name]["by_seed"][str(s)] = r
        ref = outd["reference"]["by_seed"]
        for name, d in outd.items():
            diffs = [d["by_seed"][str(s)]["reward_per_agent_tick"] - ref[str(s)]["reward_per_agent_tick"] for s in VAL]
            d["mean"] = float(np.mean([d["by_seed"][str(s)]["reward_per_agent_tick"] for s in VAL])); d["mean_diff_vs_ref"] = float(np.mean(diffs))
            d["seeds_better"] = int(sum(x > 0 for x in diffs)); d["p_signflip"] = signflip_p(diffs) if name != "reference" else None
            print(f"{name:14s} mean {d['mean']:.4f}  diff {d['mean_diff_vs_ref']:+.4f}  better {d['seeds_better']}/10  p {d['p_signflip']}", flush=True)
        json.dump(outd, open(out, "w"), indent=1)
    if mode == "es":
        kind, workers = sys.argv[3], int(sys.argv[4])
        DIM = TR.LinearRule.DIM if kind == "linear" else TR.NCARule.DIM
        T = P["training"]; NP, SIG, LR, G = T["population"], T["sigma"], T["lr"], T["generations"]
        ref_val = json.load(open(os.path.join(os.path.dirname(os.path.abspath(out)), "nca_screen.json")))["reference"]["by_seed"]
        theta = np.zeros(DIM); m_ = np.zeros(DIM); v_ = np.zeros(DIM); b1, b2 = 0.9, 0.999
        log = {"kind": kind, "dim": DIM, "generations": [], "validation": [], "best": None}
        for g in range(G):
            rng = np.random.default_rng(1000 + g)
            eps = rng.normal(size=(NP // 2, DIM)); cands = [theta + SIG * e for e in eps] + [theta - SIG * e for e in eps]
            jobs = [(V5, {"kind": kind, "theta": c.tolist()}, s, "forage", PROTO_TICKS) for c in cands for s in TRAIN]
            jobs += [(V5, {"kind": kind, "theta": theta.tolist()}, s, "forage", PROTO_TICKS) for s in TRAIN]
            res = pool_map(jobs, workers=workers, tag=f"{kind}{g}")
            fit = np.array([np.mean([res[i * len(TRAIN) + j]["reward_per_agent_tick"] for j in range(len(TRAIN))]) for i in range(NP)])
            cur = float(np.mean([r["reward_per_agent_tick"] for r in res[NP * len(TRAIN):]]))
            ranks = np.empty(NP); ranks[np.argsort(fit)] = np.arange(NP); u = ranks / (NP - 1) - 0.5
            grad = ((u[:NP // 2] - u[NP // 2:])[:, None] * eps).sum(0) / (NP * SIG)
            m_ = b1 * m_ + (1 - b1) * grad; v_ = b2 * v_ + (1 - b2) * grad ** 2
            theta = theta + LR * (m_ / (1 - b1 ** (g + 1))) / (np.sqrt(v_ / (1 - b2 ** (g + 1))) + 1e-8)
            log["generations"].append({"gen": g, "theta_train_fitness": cur, "cand_mean": float(fit.mean()), "cand_best": float(fit.max()), "theta_norm": float(np.linalg.norm(theta))})
            print(f"[{kind}] gen {g:2d}  current-theta train {cur:.4f}  candidates mean {fit.mean():.4f} best {fit.max():.4f}  |theta| {np.linalg.norm(theta):.3f}", flush=True)
            if (g + 1) % 6 == 0:
                vr = pool_map([(V5, {"kind": kind, "theta": theta.tolist()}, s, "forage", PROTO_TICKS) for s in VAL], workers=workers, tag=f"{kind}v{g}")
                diffs = [r["reward_per_agent_tick"] - ref_val[str(s)]["reward_per_agent_tick"] for r, s in zip(vr, VAL)]
                entry = {"gen": g, "theta": theta.tolist(), "val_mean": float(np.mean([r["reward_per_agent_tick"] for r in vr])), "val_mean_diff": float(np.mean(diffs)),
                         "val_better": int(sum(d > 0 for d in diffs)), "val_by_seed": vr}
                log["validation"].append(entry)
                if log["best"] is None or entry["val_mean_diff"] > log["best"]["val_mean_diff"]: log["best"] = entry
                print(f"[{kind}] validation after gen {g}: mean diff vs reference {entry['val_mean_diff']:+.4f}, better on {entry['val_better']}/10", flush=True)
            json.dump(log, open(out, "w"))
    if mode == "test":
        sel = json.load(open(sys.argv[3])); workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        arms = {"reference": {"kind": "engine"}, "no_trails": {"kind": "no_trails"}, "selected": sel["spec"]}
        outd = {"selected_name": sel["name"], "scenarios": {}}
        for sc in ("forage", "mixed"):
            jobs = [(V5, spec, s, sc, PROTO_TICKS) for spec in arms.values() for s in TEST]
            res = pool_map(jobs, workers=workers, tag="test" + sc)
            by = {name: res[i * len(TEST):(i + 1) * len(TEST)] for i, name in enumerate(arms)}
            sres = {}
            for name in ("no_trails", "selected"):
                for key in ("reward_per_agent_tick", "mean_energy"):
                    d = [a[key] - b[key] for a, b in zip(by[name], by["reference"])]
                    ref_mean = float(np.mean([b[key] for b in by["reference"]]))
                    sres[f"{name}|{key}"] = {"ref_mean": ref_mean, "arm_mean": float(np.mean([a[key] for a in by[name]])), "mean_diff": float(np.mean(d)),
                                             "rel_gain": float(np.mean(d) / ref_mean), "seeds_better": int(sum(x > 0 for x in d)), "p_signflip": signflip_p(d),
                                             "t_ci95": [float(np.mean(d) - 2.093 * np.std(d, ddof=1) / math.sqrt(len(d))), float(np.mean(d) + 2.093 * np.std(d, ddof=1) / math.sqrt(len(d)))]}
            outd["scenarios"][sc] = {"stats": sres, "by_arm": by}
            for k, v in sres.items(): print(sc, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "t_ci95"}, flush=True)
        json.dump(outd, open(out, "w"), indent=1)
    if mode == "sanity":
        jobs = []
        for seed in (20262901, 20262902):
            jobs += [(V5, {"kind": "engine"}, seed, "forage", 300),
                     (V5, {"kind": "decay"}, seed, "forage", 300), (V5, {"kind": "nca", "theta": [0.0] * TR.NCARule.DIM}, seed, "forage", 300),
                     (V5, {"kind": "linear", "theta": [0.0] * TR.LinearRule.DIM}, seed, "forage", 300)]
        res = pool_map(jobs)
        json.dump({"jobs": [[os.path.basename(j[0]), j[1]["kind"], j[2]] for j in jobs], "results": res}, open(out, "w"), indent=1)
        for j, r in zip(jobs, res): print(os.path.basename(j[0]), j[1]["kind"], j[2], r["state_hash"], r["reward_per_agent_tick"], r["seconds"], flush=True)
