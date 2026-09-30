"""Section 4.1: natural evolution, mechanism comparisons, sensitivity, and topology experiments."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import scipy
from scipy.stats import t as student_t

from sim_core import Model, simulate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
DATA = ROOT / 'data'
SEEDS = tuple(range(10))
BASE = dict(topology='BA', n=50, game='PD', benefit=5., cost=1.,
            mean=None, spread=None, epsilon=.5, beta=.8, alpha=.5,
            mechanism='reciprocal', normalized=False,
            rtol=1e-7, atol=1e-9, max_step=.2)


def graph_for(topology: str, n: int, seed: int):
    """Generate a graph by seed; resample disconnected ER/WS graphs."""
    if topology == 'BA':
        g = nx.barabasi_albert_graph(n, 2, seed=seed)
        attempt = 0
    elif topology == 'karate':
        g = nx.karate_club_graph()
        attempt = 0
    else:
        for attempt in range(1000):
            draw_seed = seed * 1000 + attempt
            if topology == 'ER':
                g = nx.erdos_renyi_graph(n, 6 / (n - 1), seed=draw_seed)
            elif topology == 'WS':
                g = nx.watts_strogatz_graph(n, 6, .2, seed=draw_seed)
            else:
                raise ValueError(topology)
            if nx.is_connected(g):
                break
        else:
            raise RuntimeError('No connected graph in 1000 draws')
    edges = np.asarray(sorted((min(i, j), max(i, j)) for i, j in g.edges()), dtype=int)
    return g.number_of_nodes(), edges, attempt


def initial_state(n: int, seed: int, mean: float, spread: float):
    """Center uniform samples at the specified mean with maximum deviation spread."""
    rng = np.random.default_rng(10000 + seed)
    if mean is None:
        return rng.uniform(0, 1, n)
    z = rng.uniform(-1, 1, n)
    z -= z.mean()
    return mean + spread * z / np.max(np.abs(z))


def payoff(game, benefit, cost):
    if game == 'PD':
        return np.array([[benefit - cost, -cost], [benefit, 0.]])
    return np.array([[benefit - cost / 2, benefit - cost], [benefit, 0.]])


def make_model(c, n, edges):
    args = dict(n=n, edges=edges, A=payoff(c['game'], c['benefit'], c['cost']),
                epsilon=c['epsilon'], beta=c['beta'], alpha=c['alpha'])
    reference = Model(**args)
    if c['normalized']:
        # Match total interaction strength at the same state.
        model = Model(**args, mechanism=c['mechanism'],
                      normalize_strength=lambda x: 2 * reference.quantities(x)['w_ij'].sum())
    else:
        model = Model(**args, mechanism=c['mechanism'])
    return model, reference


def case_key(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def all_configs(seeds):
    result = {}
    def add(study, **kwargs):
        c = dict(BASE, **kwargs)
        for seed in seeds:
            actual = dict(c, seed=int(seed))
            key = case_key(actual)
            if key not in result:
                result[key] = dict(config=actual, studies=[])
            result[key]['studies'].append(study)
    for game in ('PD', 'SD'):
        for mean in (.3, .7):
            for benefit in (2., 5.):
                add('initial_mean', game=game, mean=mean, spread=.2, benefit=benefit)
        for mechanism, normalized in (('reciprocal', False), ('unbiased', False),
                                      ('original', False), ('unbiased', True),
                                      ('original', True)):
            add('mechanism', game=game, mechanism=mechanism, normalized=normalized)
    for topology, n in (('BA', 50), ('BA', 200), ('BA', 500), ('ER', 50),
                        ('WS', 50), ('karate', 34)):
        add('topology', topology=topology, n=n)
    for parameter, values in (('epsilon', (.25, .5, 1.)), ('beta', (.2, .8, 2.)),
                              ('alpha', (.3, .5, .7))):
        for value in values:
            add('sensitivity_' + parameter, **{parameter: value})
    return [(key, data['config'], sorted(set(data['studies'])))
            for key, data in result.items()]


def execute(task):
    key, c, studies = task
    n, edges, graph_attempt = graph_for(c['topology'], c['n'], c['seed'])
    x0 = initial_state(n, c['seed'], c['mean'], c['spread'])
    model, reference = make_model(c, n, edges)
    start = time.perf_counter()
    out = simulate(model, x0, t_end=400., max_step=c['max_step'],
                   rtol=c['rtol'], atol=c['atol'], sample_dt=.2,
                   arrival_tol=1e-3, stop_at_arrival=True)
    elapsed = time.perf_counter() - start
    states = np.asarray(out['x'])
    if states.shape[0] != len(out['t']):
        states = states.T
    means = states.mean(axis=1)
    initial_strength = float(2 * model.quantities(x0)['w_ij'].sum())
    ref_strength = float(2 * reference.quantities(x0)['w_ij'].sum())
    numerical = dict(max_mean_drift=float(np.max(np.abs(means-x0.mean()))),
                     min_state=float(states.min()), max_state=float(states.max()),
                     final_error=float(np.max(np.abs(states[-1] - x0.mean()))),
                     rhs_sum_initial=float(abs(model.natural(x0).sum())))
    # The time bound in Section 2.3 applies only to the reciprocal model.
    bound = reference.consensus_bound(x0) if c['mechanism'] == 'reciprocal' else None
    if isinstance(bound, dict):
        bound = bound.get('bound', bound.get('time_bound', bound.get('T')))
    row = dict(c, case_id=key, studies=';'.join(studies), n=n,
               edges=len(edges), graph_attempt=graph_attempt,
               arrival=float(out['arrival']) if out['arrival'] is not None else None,
               reached=out['arrival'] is not None,
               theoretical_bound=float(bound) if bound is not None else None,
               initial_mean=float(x0.mean()), final_mean=float(means[-1]),
               initial_strength=initial_strength, reference_initial_strength=ref_strength,
               elapsed_seconds=elapsed, nfev=int(out['nfev']), **numerical)
    if c['seed'] == SEEDS[0]:
        # Use the first seed for representative trajectories.
        trace = dict(case_id=key, config=c, studies=studies,
                     t=np.asarray(out['t']).tolist(), x=states.tolist(),
                     error=np.max(np.abs(states-x0.mean()),axis=1).tolist(),
                     mean=means.tolist(), arrival=row['arrival'],
                     theoretical_bound=row['theoretical_bound'])
        (RESULTS / f'natural_trace_{key}.json').write_text(json.dumps(trace), encoding='utf-8')
    return row


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean_ci(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    mean = float(x.mean()) if len(x) else None
    half = float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
    return mean, half


def summaries(rows):
    groups = {}
    fields = [k for k in BASE if k not in ('rtol', 'atol', 'max_step')]
    for row in rows:
        for study in row['studies'].split(';'):
            key = (study,)+tuple(row[k] for k in fields)
            groups.setdefault(key, []).append(row)
    summary=[]
    for key, samples in groups.items():
        entry=dict(study=key[0], **dict(zip(fields,key[1:])), trials=len(samples),
                   reached=sum(r['reached'] for r in samples))
        for metric in ('arrival','theoretical_bound','elapsed_seconds','nfev','initial_strength'):
            mean, half=mean_ci([r[metric] for r in samples])
            entry[metric+'_mean']=mean
            entry[metric+'_ci95_half']=half
        entry['max_mean_drift']=max(r['max_mean_drift'] for r in samples)
        entry['min_state']=min(r['min_state'] for r in samples)
        entry['max_state']=max(r['max_state'] for r in samples)
        summary.append(entry)
    return summary


def paired_comparisons(rows):
    """Pair runs by seed using the same graph and initial state; report arrival-time differences."""
    records=[]
    for game in ('PD','SD'):
        subset=[r for r in rows if 'mechanism' in r['studies'].split(';') and r['game']==game]
        reference={r['seed']:r for r in subset if r['mechanism']=='reciprocal'}
        for mechanism,normalized in (('original',False),('unbiased',False),('original',True),('unbiased',True)):
            samples=[r for r in subset if r['mechanism']==mechanism and r['normalized']==normalized]
            differences=[r['arrival']-reference[r['seed']]['arrival'] for r in samples
                         if r['arrival'] is not None and reference[r['seed']]['arrival'] is not None]
            mean,half=mean_ci(differences)
            records.append(dict(game=game,mechanism=mechanism,normalized=normalized,
                                paired_trials=len(differences),arrival_difference_mean=mean,
                                arrival_difference_ci95_half=half,
                                interpretation='ablation arrival minus reciprocal arrival'))
    return records


def source_record():
    DATA.mkdir(exist_ok=True)
    _, edges, _=graph_for('karate',34,0)
    file=DATA/'karate_unweighted_edges.csv'
    write_csv(file,[dict(i=int(i), j=int(j)) for i,j in edges])
    info=dict(dataset='Zachary karate club', nodes=34, undirected_edges=78,
              retrieved_from='NetworkX '+nx.__version__+' nx.karate_club_graph()',
              documentation='https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html',
              original_article='Wayne W. Zachary. An Information Flow Model for Conflict and Fission in Small Groups. Journal of Anthropological Research 33(4), 452–473 (1977).',
              doi='https://doi.org/10.1086/jar.33.4.3629752',
              online_verification='Official NetworkX documentation and publisher DOI page opened successfully on 2026-09-27.',
              preprocessing='Discard empirical edge weights and club labels; keep the full connected, simple, undirected graph with zero-based labels. Model interaction weights are recomputed from states. No observations of cooperation strategies are fitted.',
              sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (DATA/'karate_source.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--accuracy',action='store_true')
    args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    source_record()
    tasks=all_configs(SEEDS[:1] if args.smoke else SEEDS)
    if args.accuracy:
        tasks=[]
        for topology,n in (('BA',50),('BA',500),('karate',34)):
            c=dict(BASE,topology=topology,n=n,seed=0,rtol=2e-8,atol=2e-10,max_step=.1)
            tasks.append((case_key(c),c,['accuracy']))
    path=RESULTS/('natural_accuracy.jsonl' if args.accuracy else 'natural_runs.jsonl')
    existing=[json.loads(line) for line in path.read_text().splitlines()] if args.resume and path.exists() else []
    done={r['case_id'] for r in existing}
    tasks=[task for task in tasks if task[0] not in done]
    rows=list(existing)
    metadata=dict(python=platform.python_version(), numpy=np.__version__,scipy=scipy.__version__,
                  networkx=nx.__version__,platform=platform.platform(),cpu=platform.processor(),
                  seeds=list(SEEDS),base=BASE,time_horizon=400.,sample_dt=.2,arrival_tol=1e-3,
                  graph_parameters=dict(BA_m=2,ER_p='6/(N-1), rejection conditioned on connectivity',WS_k=6,WS_p=.2),
                  initial_states='Baseline: rng(10000+seed).uniform(0,1,N). Mean experiment: uniform(-1,1,N), subtract sample mean, divide maximum absolute entry, multiply spread=.2 and add mean=.3 or .7',
                  normalization='w_ablation(x) *= sum_edges w_reciprocal(x) / sum_edges w_ablation(x); thus total strength agrees at every common state x',
                  ci='mean +/- t_(0.975,n-1) * sample standard deviation / sqrt(n)',
                  workers=args.workers)
    (RESULTS/('natural_accuracy_metadata.json' if args.accuracy else 'natural_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    with path.open('a' if args.resume else 'w') as f:
        if args.workers>1:
            executor=ProcessPoolExecutor(args.workers)
            iterator=executor.map(execute,tasks)
        else:
            executor=None
            iterator=map(execute,tasks)
        for i,row in enumerate(iterator,1):
            rows.append(row)
            f.write(json.dumps(row)+'\n'); f.flush()
            print(f'{i}/{len(tasks)} {row["studies"]} {row["topology"]}{row["n"]} {row["game"]} seed={row["seed"]} t={row["arrival"]} runtime={row["elapsed_seconds"]:.3f}',flush=True)
        if executor:
            executor.shutdown()
    prefix='natural_accuracy' if args.accuracy else 'natural'
    write_csv(RESULTS/(prefix+'_runs.csv'),rows)
    write_csv(RESULTS/(prefix+'_summary.csv'),summaries(rows))
    if not args.accuracy:
        write_csv(RESULTS/'natural_paired_comparisons.csv',paired_comparisons(rows))
    print(json.dumps(dict(runs=len(rows),reached=sum(r['reached'] for r in rows),
                          max_mean_drift=max(r['max_mean_drift'] for r in rows)),indent=2))


if __name__=='__main__':
    main()
