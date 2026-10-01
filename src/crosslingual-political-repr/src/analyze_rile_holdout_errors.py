"""Describe category-level errors and paired prediction disagreements."""
import argparse
import gzip
import hashlib
import html
import json
import platform
import time
from pathlib import Path

import numpy as np

from bootstrap_rile_transfer import LANGS
from evaluate_rile_category_holdout import DATA, PINS, artifact, load_cohorts, records, sha

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOLDOUT = ROOT / 'gcp-workspace/rile_v2_category_holdout/full-batch8-20260930-luna'
DEFAULT_BOOTSTRAP = ROOT / 'gcp-workspace/rile_v2_category_holdout/bootstrap-batch8-20260930-luna/bootstrap_results.json'
CATEGORIES = ('107', '506', '601', '603')
MODELS = tuple(PINS)


def interval(point, draws):
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {'point': np.asarray(point).tolist(), 'low': low.tolist(), 'high': high.tolist()}


def category_rate_draws(weights, cluster_correct, denominators):
    valid = denominators > 0
    return (weights @ cluster_correct)[valid] / denominators[valid] * 100, valid


def category_draw_fixture():
    weights = np.array([[1, 0], [0, 1], [0, 0]])
    numerators = np.array([1, 0])
    denominators = np.array([1, 1])
    draws, valid = category_rate_draws(weights, numerators, weights @ denominators)
    ci = interval(50, draws)
    assert valid.tolist() == [True, True, False]
    assert draws.tolist() == [100, 0]
    assert ci['low'] == 2.5 and ci['high'] == 97.5


def read_predictions(path, expected, layers):
    cells = {(s, t): {} for s in LANGS for t in LANGS}
    for row in records(path):
        item = row['item_id']
        assert item in expected
        source, target = row['source_language'], row['target_language']
        assert (row['manifesto_id'], row['fold'], row['true_label'], row['code']) == expected[item]
        assert row['layer_index'] == layers[row['fold'], source]
        cell = cells[source, target]
        assert item not in cell
        cell[item] = int(row['predicted_label'])
    assert all(set(cell) == set(expected) for cell in cells.values())
    return cells


def bootstrap_weights(hold, matched_dev, manifestos, draws, seed):
    cluster = {m: i for i, m in enumerate(manifestos)}
    totals = np.zeros((len(manifestos), 2), dtype=np.int64)
    dev_totals = np.zeros_like(totals)
    for row in hold:
        totals[cluster[row['manifesto_id']], row['label']] += 1
    for row in matched_dev:
        dev_totals[cluster[row['manifesto_id']], row['label']] += 1
    rng = np.random.default_rng(seed)
    weights = np.empty((0, len(manifestos)), dtype=np.int64)
    discarded = 0
    while len(weights) < draws:
        batch = rng.multinomial(len(manifestos), np.full(len(manifestos), 1 / len(manifestos)), size=draws - len(weights))
        valid = ((batch @ totals) > 0).all(1) & ((batch @ dev_totals) > 0).all(1)
        discarded += int((~valid).sum())
        weights = np.concatenate([weights, batch[valid]])
    digest = hashlib.sha256(weights.tobytes()).hexdigest()
    return weights, totals, digest, discarded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument('--bootstrap', type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.monotonic()
    dev, hold, foldmap = load_cohorts()
    manifestos = sorted({r['manifesto_id'] for r in hold})
    cluster = {m: i for i, m in enumerate(manifestos)}
    matched_dev = [r for r in dev if r['manifesto_id'] in cluster]
    weights, totals, weights_sha, discarded = bootstrap_weights(hold, matched_dev, manifestos, 10000, 20260930)
    upstream = json.loads(args.bootstrap.read_text())
    method = upstream['method']
    assert weights_sha == method['weights_sha256'] and discarded == method['discarded_zero_class_draws'] == 0
    assert method['draws'] == len(weights) and method['seed'] == 20260930
    category_draw_fixture()

    item_rows = {r['item_id']: r for r in hold}
    expected = {r['item_id']: (r['manifesto_id'], foldmap[r['manifesto_id']], r['label'], r['code']) for r in hold}
    by_model = {}
    provenance = {}
    for model in MODELS:
        folder = args.holdout / model
        summary_path = folder / 'summary.json'
        summary = json.loads(summary_path.read_text())
        pred_path = folder / f'{model}_predictions.jsonl.gz'
        assert summary['status'] == 'complete' and summary['input']['sha256'] == sha(DATA)
        assert summary['foldmap'] == foldmap and summary['cohort_items'] == len(hold)
        assert sha(pred_path) == summary['prediction_files'][model]['sha256']
        layers = {(r['fold'], r['source_language']): r['layer_index'] for r in summary['selected_layers']}
        assert set(layers) == {(f, s) for f in range(5) for s in LANGS}
        cells = read_predictions(pred_path, expected, layers)
        # Recompute each original balanced-accuracy point before category slicing.
        matrix = upstream['families'][model]['holdout']['matrix']['point']
        for si, source in enumerate(LANGS):
            for ti, target in enumerate(LANGS):
                correct = [int(cells[source, target][r['item_id']] == r['label']) for r in hold]
                recall = [np.mean([v for v, r in zip(correct, hold) if r['label'] == label]) for label in (0, 1)]
                point = float(np.mean(recall)) * 100
                assert abs(point - matrix[si][ti]) < 1e-12
                assert abs(point / 100 - summary['matrices'][model][source][target]['balanced_accuracy']) < 1e-12
        by_model[model] = cells
        provenance[model] = {'prediction_sha256': sha(pred_path), 'summary_sha256': sha(summary_path),
                             'selected_layers': summary['selected_layers'], 'model': summary['model'],
                             'model_revision': summary['model_revision']}

    cat_rows = {code: [r for r in hold if r['code'] == code] for code in CATEGORIES}
    assert {k: len(v) for k, v in cat_rows.items()} == {'107': 287, '506': 306, '601': 271, '603': 223}
    results = {}
    category_masks = {}
    for code, rows in cat_rows.items():
        indexes = np.array([cluster[r['manifesto_id']] for r in rows])
        den = weights[:, indexes].sum(1)
        mask = den > 0
        category_masks[code] = mask
        assert mask.any()
        per_model = {}
        for model in MODELS:
            good = np.zeros((6, 6), dtype=np.int64)
            draws = np.zeros((int(mask.sum()), 6, 6), dtype=float)
            cross_wrong, cross_total = 0, 0
            for si, source in enumerate(LANGS):
                for ti, target in enumerate(LANGS):
                    ok = np.array([int(by_model[model][source, target][r['item_id']] == r['label']) for r in rows])
                    good[si, ti] = int(ok.sum())
                    cluster_good = np.bincount(indexes, weights=ok, minlength=len(manifestos))
                    cell_draws, cell_mask = category_rate_draws(weights, cluster_good, den)
                    assert np.array_equal(cell_mask, mask)
                    draws[:, si, ti] = cell_draws
                    if si != ti:
                        cross_wrong += len(rows) - int(ok.sum()); cross_total += len(rows)
            points = good / len(rows) * 100
            agg = {'all_cell_mean': interval(points.mean(), draws.mean((1, 2))),
                   'diagonal_mean': interval(np.diag(points).mean(), np.diagonal(draws, axis1=1, axis2=2).mean(1)),
                   'cross_language_mean': interval((points.sum() - np.trace(points)) / 30,
                                                     (draws.sum((1, 2)) - np.diagonal(draws, axis1=1, axis2=2).sum(1)) / 30)}
            per_model[model] = {'correct_fraction_percent': interval(points, draws), 'aggregates': agg,
                                'cross_errors': cross_wrong, 'cross_predictions': cross_total}
        cross_errors_all = sum(per_model[m]['cross_errors'] for m in MODELS)
        per_model['category_item_count'] = len(rows)
        per_model['category_share_percent'] = len(rows) / len(hold) * 100
        per_model['valid_category_draws'] = int(mask.sum())
        per_model['missing_category_draws'] = int((~mask).sum())
        per_model['valid_draw_mask_sha256'] = hashlib.sha256(mask.tobytes()).hexdigest()
        for model in MODELS:
            per_model[model]['share_of_model_cross_errors_percent'] = None  # Filled after all category denominators are known.
        results[code] = per_model

    # Error-share denominator covers all 1,087 items x 30 cross-language cells per model.
    for model in MODELS:
        total_wrong = 0
        for source in LANGS:
            for target in LANGS:
                if source != target:
                    total_wrong += sum(by_model[model][source, target][r['item_id']] != r['label'] for r in hold)
        for code in CATEGORIES:
            results[code][model]['share_of_model_cross_errors_percent'] = results[code][model]['cross_errors'] / total_wrong * 100

    # Item-level pairwise disagreement and exhaustive four-model correctness partitions.
    pairs = {}
    hold_cluster_counts = np.bincount([cluster[r['manifesto_id']] for r in hold], minlength=len(manifestos))
    for i, first in enumerate(MODELS):
        for second in MODELS[i + 1:]:
            cells = np.zeros((6, 6), dtype=float)
            draws = np.zeros((len(weights), 6, 6), dtype=float)
            for si, source in enumerate(LANGS):
                for ti, target in enumerate(LANGS):
                    values = np.array([by_model[first][source, target][r['item_id']] != by_model[second][source, target][r['item_id']] for r in hold], dtype=np.int64)
                    cells[si, ti] = values.mean() * 100
                    cluster_count = np.bincount([cluster[r['manifesto_id']] for r in hold], weights=values, minlength=len(manifestos))
                    draws[:, si, ti] = weights @ cluster_count / (weights @ hold_cluster_counts) * 100
            pairs[f'{first} vs {second}'] = {'cell_disagreement_percent': interval(cells, draws),
                'diagonal_mean': interval(np.diag(cells).mean(), np.diagonal(draws, axis1=1, axis2=2).mean(1)),
                'cross_language_mean': interval((cells.sum() - np.trace(cells)) / 30,
                    (draws.sum((1, 2)) - np.diagonal(draws, axis1=1, axis2=2).sum(1)) / 30)}

    partition_names = ('unanimous_correct', 'unanimous_wrong', 'three_to_one', 'two_to_two')
    partitions = {c: dict.fromkeys(partition_names, 0) for c in CATEGORIES}
    partition_cells = {c: {name: np.zeros((6, 6), dtype=np.int64) for name in partition_names} for c in CATEGORIES}
    candidates = []
    for si, source in enumerate(LANGS):
        for ti, target in enumerate(LANGS):
            for row in hold:
                votes = [int(by_model[m][source, target][row['item_id']] == row['label']) for m in MODELS]
                ncorrect = sum(votes)
                key = ('unanimous_correct' if ncorrect == 4 else 'unanimous_wrong' if ncorrect == 0 else
                       'three_to_one' if ncorrect in (1, 3) else 'two_to_two')
                partitions[row['code']][key] += 1
                partition_cells[row['code']][key][si, ti] += 1
                if ncorrect != 4:
                    candidates.append({'item_id': row['item_id'], 'manifesto_id': row['manifesto_id'],
                        'category': row['code'], 'fold': foldmap[row['manifesto_id']], 'source_language': source,
                        'target_language': target, 'model_correctness': dict(zip(MODELS, votes))})
    partition_details = {}
    for code, stats in partitions.items():
        assert sum(stats.values()) == len(cat_rows[code]) * 36
        matrices = {name: values.tolist() for name, values in partition_cells[code].items()}
        diagonal = {name: int(np.trace(values)) for name, values in partition_cells[code].items()}
        cross = {name: int(values.sum() - np.trace(values)) for name, values in partition_cells[code].items()}
        assert all(int(values.sum()) == stats[name] for name, values in partition_cells[code].items())
        assert sum(diagonal.values()) == len(cat_rows[code]) * 6
        assert sum(cross.values()) == len(cat_rows[code]) * 30
        assert sum(diagonal.values()) + sum(cross.values()) == sum(stats.values())
        for si in range(6):
            for ti in range(6):
                assert sum(values[si, ti] for values in partition_cells[code].values()) == len(cat_rows[code])
        partition_details[code] = {'cells': matrices, 'diagonal_totals': diagonal, 'cross_language_totals': cross}

    audit = args.output / 'manual_review_candidates.jsonl.gz'
    args.output.mkdir(parents=True)
    with gzip.open(audit, 'wt') as stream:
        for row in candidates:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    result = {'method': {'date': '2026-09-30', 'metric': 'fraction correct within each single-label category; no within-category balanced accuracy',
        'bootstrap': '10,000 shared 58-manifesto weights, seed 20260930; category interval uses only draws with category items; shared masks across models/pairs',
        'interval': 'pointwise 95% percentile; not multiplicity adjusted', 'holdout_items': len(hold),
        'class_counts': [593, 494], 'category_counts': {k: len(v) for k, v in cat_rows.items()},
        'languages': LANGS, 'cell_count': 36, 'cross_language_cells': 30,
        'limitations': 'Repeated item-language predictions are not independent items. Results condition on frozen folds, probes and selected layers; no category-population, ideology, party, significance, or causal inference.'},
        'provenance': {'input_sha256': sha(DATA), 'code_sha256': sha(__file__), 'helpers': {
            'evaluate_rile_category_holdout.py': sha(Path(__file__).with_name('evaluate_rile_category_holdout.py')),
            'bootstrap_rile_category_holdout.py': sha(Path(__file__).with_name('bootstrap_rile_category_holdout.py')),
            'bootstrap_rile_transfer.py': sha(Path(__file__).with_name('bootstrap_rile_transfer.py'))},
            'upstream_bootstrap_sha256': sha(args.bootstrap), 'upstream_weights_sha256': weights_sha, 'models': provenance},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__}, 'categories': results,
        'balanced_accuracy_reference': {m: upstream['families'][m]['holdout']['matrix'] for m in MODELS},
        'pairwise_model_disagreement': pairs, 'four_model_correctness_partitions': partitions,
        'four_model_correctness_partitions_by_category': partition_details,
        'manual_review_candidates': {'path': audit.name, 'sha256': sha(audit), 'rows': len(candidates), 'contains_text': False}}
    result['runtime']['elapsed_seconds'] = round(time.monotonic() - started, 2)
    (args.output / 'error_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    html_rows = ['<!doctype html><meta charset="utf-8"><title>RILE category error analysis</title><style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse;margin:14px 0}th,td{padding:7px;border:1px solid #bbb}th{background:#eee}</style>',
      '<h1>Frozen RILE-v2 category error analysis</h1><p>Cells show fraction correct (%) with pointwise 95% manifesto-bootstrap intervals. A category has one label, so its metric is recall/fraction correct. Intervals use only draws containing that category and share the same draw mask across models.</p><p>Cross-language summaries average 30 source-target cells; each item contributes 30 repeated predictions. These are not 30 independent samples. <a href="error_analysis.json">Complete results and provenance</a> · <a href="manual_review_candidates.jsonl.gz">Text-free review candidates</a></p>']
    lang_names = ['English', 'Spanish', 'German', 'Chinese', 'Hindi', 'Marathi']
    for model in MODELS:
        mat = result['balanced_accuracy_reference'][model]
        html_rows.append(f'<h2>{html.escape(model)} balanced-accuracy reference</h2><table><tr><th>Source → target</th>' + ''.join(f'<th>{n}</th>' for n in lang_names) + '</tr>')
        for si, name in enumerate(lang_names):
            cells = ''.join(f'<td>{mat["point"][si][ti]:.2f} [{mat["low"][si][ti]:.2f}, {mat["high"][si][ti]:.2f}]</td>' for ti in range(6))
            html_rows.append(f'<tr><th>{name}</th>{cells}</tr>')
        html_rows.append('</table>')
    for code in CATEGORIES:
        for model in MODELS:
            mat = results[code][model]['correct_fraction_percent']
            html_rows.append(f'<h2>Category {code}: {html.escape(model)} fraction correct</h2><table><tr><th>Source → target</th>' + ''.join(f'<th>{n}</th>' for n in lang_names) + '</tr>')
            for si, name in enumerate(lang_names):
                cells = ''.join(f'<td>{mat["point"][si][ti]:.2f} [{mat["low"][si][ti]:.2f}, {mat["high"][si][ti]:.2f}]</td>' for ti in range(6))
                html_rows.append(f'<tr><th>{name}</th>{cells}</tr>')
            html_rows.append('</table>')
    for model in MODELS:
        html_rows.append(f'<h2>{html.escape(model)}</h2><table><tr><th>Category</th><th>Items / share</th><th>Cross-language correct % [95% interval]</th><th>Cross errors / predictions</th><th>Share of model cross errors</th></tr>')
        for code in CATEGORIES:
            cat = results[code]; metric = cat[model]['aggregates']['cross_language_mean'];
            html_rows.append(f'<tr><th>{code}</th><td>{cat["category_item_count"]} / {cat["category_share_percent"]:.1f}%</td><td>{metric["point"]:.2f} [{metric["low"]:.2f}, {metric["high"]:.2f}]</td><td>{cat[model]["cross_errors"]} / {cat[model]["cross_predictions"]}</td><td>{cat[model]["share_of_model_cross_errors_percent"]:.1f}%</td></tr>')
        html_rows.append('</table>')
    html_rows.append('<h2>Paired model disagreement</h2><table><tr><th>Pair</th><th>Cross-language disagreement % [95% interval]</th><th>Diagonal disagreement % [95% interval]</th></tr>')
    for pair, data in pairs.items():
        x, d = data['cross_language_mean'], data['diagonal_mean']
        html_rows.append(f'<tr><th>{html.escape(pair)}</th><td>{x["point"]:.2f} [{x["low"]:.2f}, {x["high"]:.2f}]</td><td>{d["point"]:.2f} [{d["low"]:.2f}, {d["high"]:.2f}]</td></tr>')
    html_rows.append('</table>')
    for pair, data in pairs.items():
        mat = data['cell_disagreement_percent']
        html_rows.append(f'<h3>{html.escape(pair)} cell disagreement</h3><table><tr><th>Source → target</th>' + ''.join(f'<th>{n}</th>' for n in lang_names) + '</tr>')
        for si, name in enumerate(lang_names):
            cells = ''.join(f'<td>{mat["point"][si][ti]:.2f} [{mat["low"][si][ti]:.2f}, {mat["high"][si][ti]:.2f}]</td>' for ti in range(6))
            html_rows.append(f'<tr><th>{name}</th>{cells}</tr>')
        html_rows.append('</table>')
    html_rows.append('<h2>Four-model correctness partitions</h2><p>Cell matrices and all-cell, diagonal, and cross-language totals appear in the JSON.</p><table><tr><th>Category</th><th>Partition</th><th>All 36 cells</th><th>Diagonal 6</th><th>Cross-language 30</th></tr>')
    for code in CATEGORIES:
        detail = partition_details[code]
        for name in partition_names:
            html_rows.append(f'<tr><th>{code}</th><td>{name}</td><td>{partitions[code][name]}</td><td>{detail["diagonal_totals"][name]}</td><td>{detail["cross_language_totals"][name]}</td></tr>')
    html_rows.append('</table>')
    (args.output / 'tables.html').write_text('\n'.join(html_rows) + '\n')
    print(json.dumps({'output': str(args.output), 'elapsed_seconds': result['runtime']['elapsed_seconds'],
        'weights_sha256': weights_sha, 'discarded_draws': discarded, 'candidate_rows': len(candidates),
        'category_cross_accuracy': {c: {m: results[c][m]['aggregates']['cross_language_mean'] for m in MODELS} for c in CATEGORIES},
        'pair_cross_disagreement': {p: v['cross_language_mean'] for p, v in pairs.items()}}, indent=2))


if __name__ == '__main__':
    main()
