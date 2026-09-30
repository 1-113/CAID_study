"""Section 4.2: independent tuning, controller comparisons, sensitivity, and topology experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import networkx as nx
import numpy as np
import scipy
from scipy.stats import t as student_t

from sec4_2_control import ControlConfig, run_control, paired_problem
from sim_core import simulate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEEDS = list(range(10))
TUNE_SEEDS = [100, 101, 102]
RESUME = False


def finite_mean_ci(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    m = float(arr.mean())
    h = float(student_t.ppf(.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.
    return {"n": len(arr), "mean": m, "ci95": [m - h, m + h]}


def summarize(records, group_keys):
    groups = {}
    for row in records:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, rows in groups.items():
        entry = {**dict(zip(group_keys, key)), "runs": len(rows),
                 "reached": sum(r["reached"] for r in rows),
                 "violations": sum(r["state_violation"] for r in rows)}
        for metric in ["arrival_time", "stop_time", "energy_to_stop", "J_switch_to_optimal",
                       "value_initial", "square_gap_integral", "rmse_integral", "terminal_max_error", "elapsed_seconds",
                       "nfev", "cpu_seconds", "pinning_time_bound", "cost_identity_error", "mean_balance_error"]:
            entry[metric] = finite_mean_ci([r.get(metric) for r in rows])
        out.append(entry)
    return out


def write_results(group, records, group_keys):
    OUT.mkdir(exist_ok=True)
    payload = {"group": group, "records": records, "summary": summarize(records, group_keys)}
    (OUT / f"control_{group}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / f"control_{group}.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return payload


def run_one(group, seed, controller, cfg, label=""):
    filename = f"control_{group}_{label}_{cfg.topology}{cfg.n}_seed{seed}_target{cfg.target:g}_{controller}.npz"
    path = OUT / "control_trajectories" / filename
    result = None
    if RESUME and path.exists():
        with np.load(path) as cached:
            previous = json.loads(str(cached["config_json"]))
            _, x0, active, info = paired_problem(seed, cfg)
            valid = (all(previous.get(k) == v for k, v in asdict(cfg).items())
                     and np.array_equal(cached["edges"], info["edges"])
                     and np.array_equal(cached["x0"], x0)
                     and np.array_equal(cached["active"], active))
            if valid:
                result = previous
    if result is None:
        result = run_control(seed, controller, cfg, path)
    result["case"] = label or group
    print(f"{group} {label} {cfg.topology}{cfg.n} seed={seed} target={cfg.target:g} {controller}: "
          f"reached={result['reached']} T={result['stop_time']:.4f} E={result['energy_to_stop']:.6f} "
          f"J={result['J_switch_to_optimal']:.6f} seconds={result['elapsed_seconds']:.3f}", flush=True)
    return result


def tune():
    records = []
    candidates = [("linear", kp, 0.1) for kp in (.5, 1., 2.)]
    candidates += [("PI", kp, ki) for kp in (.5, 1., 2.) for ki in (.02, .1, .5)]
    ranking = []
    for controller, kp, ki in candidates:
        cfg = replace(ControlConfig(), kp=kp, ki=ki, method="LSODA", rtol=1e-6, atol=1e-8, max_step=.25)
        case = f"kp{kp:g}_ki{ki:g}"
        rows = [run_one("tune", seed, controller, cfg, case) for seed in TUNE_SEEDS]
        records.extend(rows)
        ranking.append({"controller": controller, "kp": kp, "ki": ki,
                        "failures": sum(not r["reached"] or r["state_violation"] for r in rows),
                        "mean_stop_time": float(np.mean([r["stop_time"] for r in rows])),
                        "mean_energy": float(np.mean([r["energy_to_stop"] for r in rows]))})
    chosen = {}
    for kind in ("linear", "PI"):
        best = min((r for r in ranking if r["controller"] == kind),
                   key=lambda r: (r["failures"], r["mean_stop_time"], r["mean_energy"]))
        chosen[kind] = {"kp": best["kp"], "ki": best["ki"]}
    chosen["protocol"] = "Independent BA50 pilot seeds 100,101,102, target .3; lexicographic: fewest failures, lowest mean stopping time, then input energy. Gains frozen before test seeds 0--9 and target1."
    chosen["ranking"] = ranking
    (OUT / "control_tuning_selected.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    write_results("tune", records, ["controller", "kp", "ki"])
    return chosen


def selection():
    path = OUT / "control_tuning_selected.json"
    return json.loads(path.read_text()) if path.exists() else tune()


def main_comparison():
    chosen = selection()
    records = []
    for target in (.3, 1.):
        for seed in SEEDS:
            for controller in ("nonlinear", "linear", "PI"):
                cfg = replace(ControlConfig(), target=target, **chosen.get(controller, {}))
                records.append(run_one("main", seed, controller, cfg))
    payload = write_results("main", records, ["target", "controller"])
    paired = []
    for target in (.3, 1.):
        for base in ("linear", "PI"):
            for metric in ("arrival_time", "energy_to_stop", "J_switch_to_optimal"):
                diffs = []
                for seed in SEEDS:
                    a = next(r for r in records if r["seed"] == seed and r["target"] == target and r["controller"] == "nonlinear")
                    b = next(r for r in records if r["seed"] == seed and r["target"] == target and r["controller"] == base)
                    if a[metric] is not None and b[metric] is not None:
                        diffs.append(a[metric] - b[metric])
                paired.append({"target": target, "baseline": base, "metric": metric,
                               "difference": "nonlinear minus baseline", **finite_mean_ci(diffs)})
    (OUT / "control_paired_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    return payload


def sensitivity():
    records = []
    # Fix the input limit at .5; compare only time and energy across gains/actuator sets.
    for gain in (.125, .25, .5):
        for seed in SEEDS:
            cfg = replace(ControlConfig(), target=.3, gain=gain)
            records.append(run_one("sensitivity", seed, "nonlinear", cfg, f"gain{gain:g}"))
    for count in (1, 3, 10):
        for seed in SEEDS:
            cfg = replace(ControlConfig(), target=.3, count=count)
            records.append(run_one("sensitivity", seed, "nonlinear", cfg, f"L{count}"))
    return write_results("sensitivity", records, ["case", "gain", "count"])


def topology():
    records = []
    cases = [("BA", 200, 20), ("BA", 500, 50), ("ER", 50, 5), ("WS", 50, 5), ("Karate", 34, 4)]
    for topology_name, n, count in cases:
        for seed in SEEDS:
            cfg = replace(ControlConfig(), target=.3, topology=topology_name, n=n, count=count)
            records.append(run_one("topology", seed, "nonlinear", cfg, f"{topology_name}{n}"))
    return write_results("topology", records, ["topology", "n", "count"])


def numerical():
    chosen = selection()
    records = []
    for target in (.3, 1.):
        for controller in ("nonlinear", "linear", "PI"):
            for label, rt, at, step in [("default", 1e-7, 1e-9, .1), ("tight", 1e-8, 1e-10, .05)]:
                cfg = replace(ControlConfig(), target=target, rtol=rt, atol=at, max_step=step,
                              **chosen.get(controller, {}))
                records.append(run_one("numerical", 0, controller, cfg, label))
    main_path = OUT / "control_main.json"
    if main_path.exists():
        main_rows = json.loads(main_path.read_text())["records"]
        worst = max((r for r in main_rows if r["controller"] == "PI" and r["target"] == 1.), key=lambda r: r["state_max"])
        for label, rt, at, step in [("worst-default", 1e-7, 1e-9, .1), ("worst-tight", 1e-8, 1e-10, .05)]:
            cfg = replace(ControlConfig(), target=1., rtol=rt, atol=at, max_step=step, **chosen["PI"])
            records.append(run_one("numerical", worst["seed"], "PI", cfg, label))
    payload = write_results("numerical", records, ["target", "controller", "case"])
    cross = []
    for target in (.3, 1.):
        cfg = replace(ControlConfig(), target=target)
        model, x0, active, info = paired_problem(0, cfg)
        result = simulate(model, x0, cfg.horizon, active=active, target=target, gains=.5,
                          rtol=1e-7, atol=1e-9, max_step=.1, sample_dt=.25,
                          arrival_tol=.001, stop_at_arrival=True, method="DOP853")
        baseline = next(r for r in records if r["target"] == target and r["controller"] == "nonlinear" and r["case"] == "default")
        final_cost = float(result["cost"][-1] + model.value(result["x"][-1], target))
        row = {"target": target, "method": "DOP853", "rtol": 1e-7, "atol": 1e-9, "max_step": .1,
               "arrival_time": result["arrival"], "energy_to_stop": float(result["energy"][-1]),
               "J_switch_to_optimal": final_cost, "nfev": result["nfev"], "elapsed_seconds": result["elapsed"],
               "arrival_difference_from_augmented_DOP853": float(result["arrival"] - baseline["arrival_time"]),
               "energy_difference_from_augmented_DOP853": float(result["energy"][-1] - baseline["energy_to_stop"]),
               "cost_difference_from_augmented_DOP853": final_cost - baseline["J_switch_to_optimal"]}
        cross.append(row)
        np.savez_compressed(OUT / "control_trajectories" / f"control_core_crosscheck_seed0_target{target:g}.npz", **result)
        print("Core cross-check", row, flush=True)
    (OUT / "control_core_crosscheck.json").write_text(json.dumps(cross, indent=2), encoding="utf-8")
    return payload


def write_manifest():
    contents = {
        "default": asdict(ControlConfig()), "main_seeds": SEEDS, "tuning_seeds": TUNE_SEEDS,
        "graph": "All topology generation delegates to sec4_1_natural.graph_for: BA(n,2,seed); ER(n,6/(n-1)), WS(n,6,.2), connected rejection draw seed=1000*seed+attempt.",
        "initial_states": "numpy.default_rng(seed+10000).uniform(0,1,n), identical across controllers",
        "selection": "first L nodes sorted by descending degree, then ascending id",
        "PI": {"raw": "v=Kp*(target-x_S)+Ki*z", "Kd": 0,
               "input": "u=clip(v,max(-bound,-f_S-h*x_S),min(bound,-f_S+h*(1-x_S)))",
               "integrator": "zdot=(target-x_S)+(u-v)/(Ki*tau)", "z0": 0,
               "information": "all methods have current graph and full state; PI guard also evaluates known f_S(x)",
               "guard_rate": 1., "antiwindup_tau": 1.},
        "cost": {"fixed_R": "2*I for each paired fixed-S comparison", "optimal_gain": .5,
                 "stopping": "T=min(first full-network max error <= 1e-3,200)",
                 "state_violation_monitor": "Numerical failure event at state departure exceeding 10*(rtol+atol); signed observed min/max saved without clipping.",
                 "J_reported": "integral_0^T(ell_S+u^T R u)dt+V(e(T)); infinite-horizon cost of baseline until T, followed by fixed-R optimal feedback thereafter",
                 "not_claimed": "The truncated integral is not the infinite-horizon cost of the original baseline continued forever.",
                 "energy": "integral_0^T sum(u_l^2)dt, stopping at each method's first threshold or common deadline"},
        "uncertainty": "two-sided Student-t 95% intervals over independent test seeds; paired differences use matched seeds",
        "network_source": {"Karate": "Zachary (1977), NetworkX karate_club_graph, undirected unweighted, original34nodes/78edges, no largest-component filtering needed"},
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "networkx": nx.__version__},
        "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT / "sec4_2_control.py", ROOT / "sec4_2_comparison.py", ROOT / "sim_core.py", ROOT / "sec4_1_natural.py"]}
    }
    (OUT / "control_parameters.json").write_text(json.dumps(contents, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["all", "tune", "main", "sensitivity", "topology", "numerical"], default="all")
    parser.add_argument("--resume", action="store_true", help="reuse matching parameters, graph and initial state after an interrupted run")
    args = parser.parse_args()
    RESUME = args.resume
    OUT.mkdir(exist_ok=True)
    groups = {"tune": tune, "main": main_comparison, "sensitivity": sensitivity, "topology": topology, "numerical": numerical}
    start = time.perf_counter()
    for name, action in groups.items():
        if args.group in ("all", name):
            action()
    write_manifest()
    print(f"Completed in {time.perf_counter() - start:.3f} seconds.", flush=True)
