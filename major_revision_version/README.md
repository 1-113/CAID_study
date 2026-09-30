# Reproducing the RB-CAID experiments

## Full experiment reproduction

Dependencies: Python, NumPy, SciPy, NetworkX, Matplotlib. Recorded versions: 3.13.12, 1.26.4, 1.17.1, 3.6.1, 3.10.9, respectively.

```sh
git clone https://github.com/1-113/CAID_study.git
cd CAID_study/major_revision_version
python sec4_1_natural.py --workers 2
python sec4_1_natural.py --accuracy
python sec4_1_report.py
OPENBLAS_NUM_THREADS=1 python sec4_2_comparison.py --group all
python sec4_3_recovery.py
python sec4_plot.py
```

Run the commands in order. The scripts create `data/`, `results/`, and `figure/` inside `major_revision_version/`; no precomputed data or figures are included. Once the experiments have generated `results/`, the last command redraws all figures.

## Parameters, random seeds, and baseline reuse

The shared model is implemented in `sim_core.py`.

| Parameter | Baseline and tested values |
|---|---|
| Payoffs | PD: `[[b-c,-c],[b,0]]`; SD: `[[b-c/2,b-c],[b,0]]`. Baseline `b=5,c=1`; natural tests also use `b=2` |
| Proposal scale | `epsilon=0.5`; sensitivity `{0.25,0.5,1}` |
| Payoff sensitivity | `beta=0.8`; sensitivity `{0.2,0.8,2}` |
| Signed-power exponent | `alpha=0.5`; sensitivity `{0.3,0.5,0.7}` |
| Nonlinear gain / input bound | `gain=0.5,bound=0.5`; gain sensitivity `{0.125,0.25,0.5}` |
| Controlled nodes | Five highest-degree nodes, ties by ascending ID; count sensitivity `{1,3,5,10}` |
| Target | Control: `0.3,1`; recovery: `0.3` |
| Linear / PI gains | Linear `Kp=2`; PI `Kp=Ki=0.5,Kd=0` |
| PI settings | Initial integral state `0`, feasibility-guard rate `h=1`, anti-windup time `tau=1` |
| Cost input weight | `R=2I` for controller comparisons |
| Replacement thresholds | Baseline `(delta,sigma)=(5,1)`; additional `(0,1),(2,1),(5,0),(5,0.25)` |

Formal runs use seeds `0,...,9`; representative curves use seed `0`. Baseline states are `default_rng(10000+seed).uniform(0,1,N)`. Prescribed-mean tests center a uniform `[-1,1]` vector, divide by its maximum absolute entry, multiply by `0.2`, and add mean `0.3` or `0.7`.

Natural mechanism comparisons use reciprocal, no-preference, and original pairwise-payoff weights. The latter two are tested at original scale and after multiplying by the reciprocal/alternative total-weight ratio at the current state.

Linear and PI tuning uses pilot seeds `100,101,102`, target `0.3`, `Kp={0.5,1,2}` and PI `Ki={0.02,0.1,0.5}`. Rank by failures, mean stopping time, then energy; freeze the chosen gains for formal runs. Comparisons share graphs, states, controlled sets, full-state information, input bounds, thresholds, and deadlines.

### Graph generation

| Graph | Procedure | Controlled nodes |
|---|---|---:|
| Barabasi-Albert | `nx.barabasi_albert_graph(N,2,seed)`, `N=50,200,500` | 5, 20, 50 |
| Erdos-Renyi | `nx.erdos_renyi_graph(50,6/49,draw_seed)` | 5 |
| Watts-Strogatz | `nx.watts_strogatz_graph(50,6,0.2,draw_seed)` | 5 |
| Karate club | `nx.karate_club_graph()`, 34 nodes, 78 edges | 4 |

ER/WS use draw seed `1000*seed+attempt`, redrawing until connected. Karate retains all nodes with zero-based indexing; empirical weights and club labels are discarded. The first natural-experiment command generates source and preprocessing details in `data/karate_source.json` and edges in `data/karate_unweighted_edges.csv`. Cooperation states and interventions are simulated.

### Recovery events and seeds

At time `5`, three of five actuators lose responsiveness and continue natural interactions. Compare no replacement, random replacement, degree-neighbor replacement, and the three-stage rule.

| Random operation | Generator seed |
|---|---|
| Failed-node selection | `20000+seed` |
| One-time uniform `[0,1]` state reset | `30000+seed` |
| Replacement at times 5 / 15 | `40000+seed` / `50000+seed` |

The shortage case allows two survivors and the lowest-index previously uncontrolled node, with no backfilling when availability returns at `15`. Complete outage lasts on `[5,15)`, then restarts with the lowest-index responsive node.

## Time, uncertainty, and metric definitions

Time is normalized continuous time. Integration uses SciPy `solve_ivp`.

| Suite | Method | rtol / atol | Maximum step | Saved interval | Deadline |
|---|---|---|---:|---:|---:|
| Natural | DOP853 | `1e-7 / 1e-9` | 0.2 | 0.2 | 400 |
| Control | DOP853 | `1e-7 / 1e-9` | 0.1 | 0.25 | 200 |
| Recovery | DOP853 | `1e-6 / 1e-8` | 0.2 | 0.1 | 300 |
| Tuning | LSODA | `1e-6 / 1e-8` | 0.25 | 0.25 | 200 |

Arrival is the first event with maximum node error `<=1e-3`, relative to the initial mean for natural runs and the target for control. Stop at arrival or deadline; retain unreached runs in stopping-time and energy summaries.

Energy integrates `sum(u**2)`; tracking error integrates mean squared node error (`rmse_integral` stores integrated MSE). Reported cost is the constructed state/input cost to stopping plus terminal value `V`. Recovery duration starts at `5`, or `15` after outage; energy and error integrals start at `0`.

Error bars are Student-t 95% confidence-interval half-widths across seeds: `t(0.975,n-1)*sample_sd/sqrt(n)`. Paired comparisons use same-seed differences.

## Generated outputs and manuscript mapping

`sec4_plot.py` generates all six PDFs from saved results.

| Section | Experiment scripts | Figures |
|---|---|---|
| 4.1 | `sec4_1_natural.py`, `sec4_1_report.py` | Figs. 1-2 |
| 4.2 | `sec4_2_control.py`, `sec4_2_comparison.py` | Figs. 3-5 |
| 4.3 | `sec4_3_recovery.py` | Fig. 6 |

Table 1 lists baseline parameters. Tables 2-4 use `control_main.json`; `natural_summary.csv`, `control_main.json`, and `control_topology.json`; `fault_results.json`, respectively.
