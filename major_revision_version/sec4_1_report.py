"""Section 4.1: reconstruct natural-evolution statistics from saved records."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from sec4_1_natural import BASE, RESULTS, paired_comparisons, summaries, write_csv


def read_records(name):
    path = RESULTS / name
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row['case_id'] for row in rows}) != len(rows):
        raise ValueError(f'{name}: empty records or duplicate case identifiers')
    return rows


def bound_comparison(rows):
    paired = [row for row in rows if row['theoretical_bound'] is not None
              and row['arrival'] is not None]
    ratios = [row['theoretical_bound'] / row['arrival'] for row in paired]
    return dict(reached_current_runs=len(paired),
                bound_min=min(row['theoretical_bound'] for row in paired),
                bound_max=max(row['theoretical_bound'] for row in paired),
                bound_to_threshold_ratio_min=min(ratios),
                bound_to_threshold_ratio_max=max(ratios))


def main():
    rows = read_records('natural_runs.jsonl')
    accuracy = read_records('natural_accuracy.jsonl')
    identity_fields = tuple(key for key in BASE if key not in ('rtol', 'atol', 'max_step')) + ('seed',)
    baseline = {tuple(row[key] for key in identity_fields): row for row in rows}
    checks = []
    for refined in accuracy:
        base = baseline[tuple(refined[key] for key in identity_fields)]
        difference = abs(refined['arrival'] - base['arrival'])
        checks.append(dict(topology=refined['topology'], n=refined['n'], seed=refined['seed'],
                           base_arrival=base['arrival'], tight_arrival=refined['arrival'],
                           absolute_difference=difference, relative_difference=difference / base['arrival']))
    summary = summaries(rows)
    pairs = paired_comparisons(rows)
    current = [row for row in rows if row['theoretical_bound'] is not None]
    topology = [row for row in current if 'topology' in row['studies'].split(';')]
    report = dict(unique_runs=len(rows), statistics_groups=len(summary),
                  seeds=sorted({row['seed'] for row in rows}),
                  all_reached=all(row['reached'] for row in rows),
                  max_sampled_mean_drift=max(row['max_mean_drift'] for row in rows),
                  sampled_state_min=min(row['min_state'] for row in rows),
                  sampled_state_max=max(row['max_state'] for row in rows),
                  all_current_below_bound=all(row['arrival'] is not None and
                      row['arrival'] < row['theoretical_bound'] for row in current),
                  accuracy=checks, theoretical_comparison=dict(all_current=bound_comparison(current),
                      baseline_topologies=bound_comparison(topology)),
                  source_sha256={name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
                      for name in ('natural_runs.jsonl', 'natural_accuracy.jsonl')})
    write_csv(RESULTS / 'natural_runs.csv', rows)
    write_csv(RESULTS / 'natural_summary.csv', summary)
    write_csv(RESULTS / 'natural_paired_comparisons.csv', pairs)
    write_csv(RESULTS / 'natural_accuracy_runs.csv', accuracy)
    write_csv(RESULTS / 'natural_accuracy_summary.csv', summaries(accuracy))
    (RESULTS / 'natural_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
