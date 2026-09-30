"""Section 4.3: loss of control responsiveness, replacement comparisons, and supplementary recovery experiments."""
from pathlib import Path
import argparse
import csv
import json
import platform
import numpy as np
import networkx as nx
from scipy.stats import t as student_t
from sim_core import Model, simulate, replace_actuators

OUT = Path(__file__).resolve().parent / 'results'
SETTINGS = dict(n=50, m=2, epsilon=.5, beta=.8, alpha=.5, gain=.5,
                target=.3, L=5, event_time=5., restore_time=15., horizon=300.,
                rtol=1e-6, atol=1e-8, max_step=.2, sample_dt=.1,
                arrival_tol=1e-3, delta=5., sigma=1.)


def setup(seed):
    g = nx.barabasi_albert_graph(SETTINGS['n'], SETTINGS['m'], seed=seed)
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=int)
    model = Model(len(g), edges, np.array([[4., -1.], [5., 0.]]),
                  epsilon=.5, beta=.8, alpha=.5)
    x0 = np.random.default_rng(10000 + seed).uniform(0., 1., len(g))
    active = sorted(sorted(g.nodes, key=lambda i: (-g.degree[i], i))[:5])
    failed = sorted(np.random.default_rng(20000 + seed).choice(active, 3, replace=False).tolist())
    return model, x0, active, failed, edges


def terminal(v):
    return float(np.asarray(v).reshape(-1)[-1])


def segment(model, x, active, start, end, stop=True, tight=False):
    return simulate(model, x, end, active=active, target=.3, gains=.5,
                    t_start=start, rtol=1e-7 if tight else 1e-6,
                    atol=1e-9 if tight else 1e-8,
                    max_step=.1 if tight else .2, sample_dt=.1,
                    arrival_tol=1e-3, stop_at_arrival=stop)


def run_case(seed, strategy='three-stage', scenario='loss', delta=5., sigma=1., tight=False):
    model, x0, active, failed, edges = setup(seed)
    parts, event_log = [], []
    part = segment(model, x0, active, 0., 5., stop=False, tight=tight)
    parts.append((part, len(active)))
    x = part['x'][-1].copy()
    pre_error = float(np.max(abs(x - .3)))
    assert pre_error > 1e-3, '故障必须发生于任务尚未完成时'
    before = x.copy()
    if scenario == 'reset':
        # Only the supplementary experiment resets states once within [0,1]; the main experiment stays continuous.
        x[failed] = np.random.default_rng(30000 + seed).uniform(0., 1., len(failed))
    if scenario == 'outage':
        responsive = []
    elif scenario == 'shortage':
        survivors = sorted(set(active) - set(failed))
        candidate = min(set(range(model.n)) - set(active))
        responsive = survivors + [candidate]
    else:
        responsive = sorted(set(range(model.n)) - set(failed))
    selection = replace_actuators(model, x, active, responsive, delta=delta, sigma=sigma,
                                  strategy=strategy, rng=np.random.default_rng(40000 + seed))
    active = sorted(selection['active'])
    event_log.append(dict(time=5., before=before.tolist(), after=x.tolist(),
                          responsive=responsive, failed=sorted(set(parts[0][0]['active']) - set(responsive)), **selection))
    recovery_start = 5.
    recovery_x = x.copy()
    if scenario in ('outage', 'shortage'):
        part = segment(model, x, active, 5., 15., stop=False, tight=tight)
        parts.append((part, len(active)))
        x = part['x'][-1].copy()
        responsive = sorted(set(range(model.n)) - set(failed))
        selection = replace_actuators(model, x, active, responsive, delta=delta, sigma=sigma,
                                      strategy=strategy, rng=np.random.default_rng(50000 + seed))
        active = sorted(selection['active'])
        event_log.append(dict(time=15., before=x.tolist(), after=x.tolist(),
                              responsive=responsive, failed=[], **selection))
        if scenario == 'outage':
            recovery_start, recovery_x = 15., x.copy()
    part = segment(model, x, active, 15. if scenario in ('outage', 'shortage') else 5.,
                   SETTINGS['horizon'], tight=tight)
    parts.append((part, len(active)))
    arrival = part.get('arrival')
    if arrival is not None and not np.isfinite(arrival):
        arrival = None
    # Keep pre/post-event records at the same time to show resets and jumps in actuator count.
    times = np.concatenate([p['t'] for p, _ in parts])
    states = np.concatenate([p['x'] for p, _ in parts])
    controls = np.concatenate([p['u'] for p, _ in parts])
    counts = np.concatenate([np.full(len(p['t']), c) for p, c in parts])
    error = np.max(abs(states-.3), axis=1)
    energy = sum(terminal(p['energy']) for p, _ in parts)
    iae = sum(float(np.trapezoid(np.mean((p['x']-.3)**2, axis=1), p['t']))
              if hasattr(np, 'trapezoid') else float(np.trapz(np.mean((p['x']-.3)**2, axis=1), p['t']))
              for p, _ in parts)
    # Check mean balance by segment; handle resets separately so jumps are not counted as input effects.
    balances = [float(np.max(abs(p['mean_balance']))) for p, _ in parts]
    bound = model.pinned_bound(recovery_x, active, .3, .5) if active else float('inf')
    row = dict(seed=seed, strategy=strategy, scenario=scenario, delta=delta, sigma=sigma,
               arrival=arrival, recovery=None if arrival is None else arrival-recovery_start,
               recovery_start=recovery_start, recovery_bound=bound,
               energy=energy, integrated_mse=iae, final_error=float(error[-1]),
               pre_failure_error=pre_error, final_count=len(active),
               min_state=float(states.min()), max_state=float(states.max()),
               max_input=float(abs(controls).max()), mean_balance=max(balances),
               nfev=sum(p['nfev'] for p, _ in parts),
               elapsed=sum(p['elapsed'] for p, _ in parts),
               initial_active=setup(seed)[2], persistent_unavailable=failed, events=event_log)
    trajectory = dict(t=times, x=states, u=controls, error=error, count=counts, edges=edges)
    return row, trajectory


def summarize(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in groups.items():
        s = dict(zip(keys, key)); s['runs'] = len(group)
        s['reached'] = sum(r['arrival'] is not None for r in group)
        for metric in ('recovery', 'energy', 'integrated_mse', 'final_error', 'elapsed'):
            vals = np.array([r[metric] for r in group if r[metric] is not None])
            s[metric+'_mean'] = float(vals.mean()) if len(vals) else None
            s[metric+'_ci95'] = (float(student_t.ppf(.975, len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals)))
                                      if len(vals)>1 else 0.)
        out.append(s)
    return out


def json_default(v):
    if isinstance(v, (set, tuple)):
        return sorted(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def write_csv(path, rows):
    flat = [{k:v for k,v in r.items() if k not in ('events','initial_active','persistent_unavailable')} for r in rows]
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)


def write_reports(payload):
    """Reconstruct summaries from per-seed records; pair differences by seed without imputing non-arrivals."""
    rows = payload['rows']
    payload['summary'] = summarize(rows, ['scenario', 'strategy'])
    payload['sensitivity_summary'] = summarize(payload['sensitivity'], ['delta', 'sigma'])
    write_csv(OUT/'fault_runs.csv', rows + payload['sensitivity'])
    current = {r['seed']: r for r in rows
               if r['scenario'] == 'loss' and r['strategy'] == 'three-stage'}
    paired = []
    for strategy in ('none', 'random', 'degree-neighbor'):
        baseline = {r['seed']: r for r in rows
                    if r['scenario'] == 'loss' and r['strategy'] == strategy}
        for metric in ('recovery', 'energy', 'integrated_mse'):
            values = np.array([current[s][metric] - baseline[s][metric]
                               for s in sorted(current.keys() & baseline.keys())
                               if current[s][metric] is not None and baseline[s][metric] is not None])
            mean = float(values.mean()) if len(values) else None
            half = (float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1)
                          / np.sqrt(len(values))) if len(values) > 1 else None)
            paired.append(dict(difference=f'three-stage minus {strategy}', metric=metric,
                               n=len(values), mean=mean,
                               ci95_lower=None if half is None else mean-half,
                               ci95_upper=None if half is None else mean+half))
    write_csv(OUT/'fault_paired_comparisons.csv', paired)
    (OUT/'fault_results.json').write_text(json.dumps(payload, indent=2, default=json_default))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--quick', action='store_true')
    parser.add_argument('--report-only', action='store_true', help='仅从已保存原始结果重建汇总，不重跑积分')
    args = parser.parse_args(); OUT.mkdir(exist_ok=True)
    if args.report_only:
        write_reports(json.loads((OUT/'fault_results.json').read_text()))
        return
    seeds = list(range(2 if args.quick else 10))
    rows = []
    for seed in seeds:
        for strategy in ('none','random','degree-neighbor','three-stage'):
            row, trajectory = run_case(seed, strategy)
            rows.append(row)
            if seed == 0:
                np.savez_compressed(OUT / f'fault_loss_{strategy}.npz', **trajectory)
            print(f'loss {seed} {strategy}: T={row["arrival"]}, E={row["energy"]:.5f}', flush=True)
        for scenario in ('reset','shortage','outage'):
            row, trajectory = run_case(seed, scenario=scenario)
            rows.append(row)
            if seed == 0:
                np.savez_compressed(OUT / f'fault_{scenario}.npz', **trajectory)
    sensitivity = []
    for delta, sigma in ((0.,1.),(2.,1.),(5.,0.),(5.,.25)):
        for seed in seeds:
            row, _ = run_case(seed, delta=delta, sigma=sigma)
            sensitivity.append(row)
    tight, _ = run_case(0, tight=True)
    base = next(r for r in rows if r['seed']==0 and r['scenario']=='loss' and r['strategy']=='three-stage')
    checks = {k:abs(tight[k]-base[k]) for k in ('arrival','energy','integrated_mse')}
    payload = dict(settings=SETTINGS, seeds=seeds, rows=rows, sensitivity=sensitivity,
                   summary=summarize(rows,['scenario','strategy']),
                   sensitivity_summary=summarize(sensitivity,['delta','sigma']),
                   refinement=checks, python=platform.python_version())
    write_reports(payload)
    print(json.dumps(payload['summary'],indent=2))


if __name__ == '__main__':
    main()
