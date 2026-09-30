"""Section 4.2: paired graphs and initial states, controllers, and continuous-time integration with metrics."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

from sim_core import Model, signed_power
from sec4_1_natural import graph_for


@dataclass(frozen=True)
class ControlConfig:
    n: int = 50
    topology: str = "BA"
    target: float = 0.3
    count: int = 5
    gain: float = 0.5
    epsilon: float = 0.5
    beta: float = 0.8
    alpha: float = 0.5
    bound: float = 0.5
    horizon: float = 200.0
    arrival_tol: float = 1e-3
    rtol: float = 1e-7
    atol: float = 1e-9
    max_step: float = 0.1
    sample_dt: float = 0.25
    method: str = "DOP853"
    kp: float = 1.0
    ki: float = 0.1
    guard_rate: float = 1.0
    antiwindup_tau: float = 1.0


def paired_problem(seed: int, cfg: ControlConfig):
    """Controllers with the same seed share the graph, initial state, and actuator set."""
    topology_name = "karate" if cfg.topology == "Karate" else cfg.topology
    n, edges, attempt = graph_for(topology_name, cfg.n, seed)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    graph_seed = (None if cfg.topology == "Karate" else
                  seed if cfg.topology == "BA" else seed * 1000 + attempt)
    model = Model(n, edges, np.array([[4., -1.], [5., 0.]]), epsilon=cfg.epsilon, beta=cfg.beta, alpha=cfg.alpha)
    x0 = np.random.default_rng(seed + 10000).uniform(0., 1., n)
    active = np.array(sorted(g.nodes(), key=lambda i: (-g.degree(i), i))[:cfg.count], dtype=int)
    return model, x0, active, {"graph_seed": graph_seed, "edges": edges, "active": active, "n": n}


def control_input(kind, x, f, z, active, cfg):
    e = cfg.target - x[active]
    if kind == "nonlinear":
        return cfg.gain * signed_power(e, cfg.alpha), np.empty(0)
    if kind == "linear":
        return np.clip(cfg.kp * e, -cfg.bound, cfg.bound), np.empty(0)
    if kind != "PI":
        raise ValueError(kind)
    raw = cfg.kp * e + cfg.ki * z
    # Input constraints give -h*x <= dx/dt <= h*(1-x); states are not clipped.
    lo = np.maximum(-cfg.bound, -f[active] - cfg.guard_rate * x[active])
    hi = np.minimum(cfg.bound, -f[active] + cfg.guard_rate * (1. - x[active]))
    if np.any(lo > hi + 1e-8):
        raise RuntimeError("Empty feasible PI input interval")
    u = np.minimum(np.maximum(raw, lo), hi)
    dz = e + (u - raw) / (cfg.ki * cfg.antiwindup_tau)
    return u, dz


def run_control(seed: int, kind: str, cfg: ControlConfig, save: Path | None = None):
    """Stop at the network-wide threshold or time limit; the tail term assumes switching to u* with fixed R."""
    model, x0, active, info = paired_problem(seed, cfg)
    n, l = len(x0), len(active)
    nz = l if kind == "PI" else 0
    offset = n + nz
    # E, J_T, square residual, integral mean input, integral squared network error
    y0 = np.r_[x0, np.zeros(nz + 5)]
    fixed_r = 2.0
    fixed_gain = 1.0 / fixed_r

    def rhs(t, y):
        x = y[:n]
        f = model.natural(x)
        u, dz = control_input(kind, x, f, y[n:offset], active, cfg)
        dx = f.copy()
        dx[active] += u
        s = signed_power(x - cfg.target, cfg.alpha)
        ustar = -fixed_gain * s[active]
        state_cost = -2.0 * np.dot(s, f) + fixed_gain * np.dot(s[active], s[active])
        energy = np.dot(u, u)
        metrics = [energy, state_cost + fixed_r * energy,
                   fixed_r * np.dot(u - ustar, u - ustar), np.sum(u) / n,
                   np.mean((x - cfg.target) ** 2)]
        return np.r_[dx, dz, metrics]

    def arrived(t, y):
        return np.max(np.abs(y[:n] - cfg.target)) - cfg.arrival_tol
    arrived.terminal = True
    arrived.direction = -1

    def state_violation(t, y):
        return min(float(np.min(y[:n])), float(1.0 - np.max(y[:n]))) + 10. * (cfg.rtol + cfg.atol)
    state_violation.terminal = True
    state_violation.direction = -1

    start = time.perf_counter()
    cpu_start = time.process_time()
    sol = solve_ivp(rhs, (0., cfg.horizon), y0, method=cfg.method, rtol=cfg.rtol,
                    atol=cfg.atol, max_step=cfg.max_step, dense_output=True,
                    events=(arrived, state_violation))
    elapsed = time.perf_counter() - start
    cpu_elapsed = time.process_time() - cpu_start
    if not sol.success:
        raise RuntimeError(sol.message)
    stop = float(sol.t[-1])
    times = np.unique(np.r_[np.arange(0., stop, cfg.sample_dt), stop])
    yy = sol.sol(times).T
    xx = yy[:, :n]
    uu = np.array([control_input(kind, row[:n], model.natural(row[:n]), row[n:offset], active, cfg)[0]
                   for row in yy])
    e = xx - cfg.target
    value0 = float(2. / (1. + cfg.alpha) * np.sum(np.abs(x0 - cfg.target) ** (1. + cfg.alpha)))
    tail = float(2. / (1. + cfg.alpha) * np.sum(np.abs(e[-1]) ** (1. + cfg.alpha)))
    vals = sol.y[offset:, -1]
    value_path = 2. / (1. + cfg.alpha) * np.sum(np.abs(e) ** (1. + cfg.alpha), axis=1)
    identity_path = yy[:, offset + 1] + value_path - value0 - yy[:, offset + 2]
    balance_path = xx.mean(axis=1) - x0.mean() - yy[:, offset + 3]
    accepted_u = np.array([control_input(kind, row[:n], model.natural(row[:n]), row[n:offset], active, cfg)[0]
                           for row in sol.y.T])
    dense_min = min(float(xx.min()), float(sol.y[:n].min()))
    dense_max = max(float(xx.max()), float(sol.y[:n].max()))
    reached = bool(len(sol.t_events[0]))
    violated = bool(len(sol.t_events[1]))
    metrics = {"seed": int(seed), "controller": kind, **asdict(cfg), "n": n,
               "edges": int(len(info["edges"])), "graph_seed": info["graph_seed"],
               "active": active.tolist(), "reached": reached, "state_violation": violated,
               "stop_time": stop, "arrival_time": stop if reached else None,
               "initial_mean": float(x0.mean()), "terminal_max_error": float(np.max(np.abs(e[-1]))),
               "terminal_rmse": float(np.sqrt(np.mean(e[-1] ** 2))),
               "energy_to_stop": float(vals[0]), "J_truncated": float(vals[1]),
               "optimal_tail_value": tail, "J_switch_to_optimal": float(vals[1] + tail),
               "value_initial": value0, "square_gap_integral": float(vals[2]),
               "cost_identity_error": float(vals[1] + tail - value0 - vals[2]),
               "mean_balance_error": float(xx[-1].mean() - x0.mean() - vals[3]),
               "max_cost_identity_error": float(np.max(np.abs(identity_path))),
               "max_mean_balance_error": float(np.max(np.abs(balance_path))),
               "rmse_integral": float(vals[4]), "state_min": dense_min, "state_max": dense_max,
               "input_abs_max": max(float(np.max(np.abs(uu))), float(np.max(np.abs(accepted_u)))), "nfev": int(sol.nfev),
               "njev": int(sol.njev), "elapsed_seconds": float(elapsed), "cpu_seconds": float(cpu_elapsed),
               "pinning_time_bound": float(model.pinned_bound(x0, active, cfg.target, cfg.gain))}
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, t=times, x=xx, u=uu, energy=yy[:, offset], cost=yy[:, offset + 1],
                            square_gap=yy[:, offset + 2], error=np.max(np.abs(e), axis=1),
                            rmse=np.sqrt(np.mean(e ** 2, axis=1)), x0=x0, active=active, edges=info["edges"],
                            z=yy[:, n:offset], config_json=json.dumps(metrics, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    cfg = ControlConfig()
    out = Path(__file__).resolve().parent / "results" / "control_demo.npz"
    print(json.dumps(run_control(0, "nonlinear", cfg, out), indent=2))
