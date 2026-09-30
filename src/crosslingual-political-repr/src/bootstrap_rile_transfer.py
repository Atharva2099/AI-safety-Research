"""Pointwise manifesto-cluster bootstrap of saved, fixed RILE v2 probes (CPU only)."""
import argparse
import gzip
import hashlib
import html
import json
import time
from pathlib import Path

import numpy as np

LANGS = ['en', 'es', 'de', 'zh', 'hi', 'mr']
NAMES = ['English', 'Spanish', 'German', 'Chinese', 'Hindi', 'Marathi']
INPUTS = {
    'OLMo': 'rile-v2-source-matrix-replay-20260929-retrieval/olmo',
    **{m: f'rile-v2-source-matrix-resume-20260930T0840Z-retrieval/{d}/results'
       for m, d in [('Qwen3.5', 'qwen'), ('Ministral', 'ministral'), ('Gemma', 'gemma')]},
}


def balanced(correct, totals):
    """Half the recall for each class; final axis is class 0, class 1."""
    return (correct / totals).mean(axis=-1)


def interval(point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {'point': np.asarray(point).tolist(), 'low': lo.tolist(), 'high': hi.tolist()}


def fixture():
    totals = np.array([[2, 1], [1, 2]])
    correct = np.array([[1, 1], [1, 0]])
    weights = np.array([2, 0])  # The first manifesto occurs twice.
    assert balanced(correct.sum(0), totals.sum(0)) == .5
    assert balanced(weights @ correct, weights @ totals) == .75
    scores = balanced(weights @ correct, weights @ totals)
    assert scores - scores == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('gcp-workspace/rile_v2_source_matrix_bootstrap'))
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    assert args.draws > 0
    started = time.monotonic()
    fixture()
    reference, counts, metadata = None, [], {}
    for model, directory in INPUTS.items():
        folder = Path('gcp-workspace') / directory
        pred = next(folder.glob('*predictions.jsonl.gz'))
        summary_path = next(folder.glob('*summary.json'))
        summary = json.loads(summary_path.read_text())
        cells = {(s, t): {} for s in LANGS for t in LANGS}
        selected = {(x['fold'], x['source_language']): x['layer_index'] for x in summary['selected_layers']}
        with gzip.open(pred, 'rt') as stream:
            for line in stream:
                r = json.loads(line)
                key = (r['item_id'], r['manifesto_id'], r['fold'], r['true_label'])
                cell = cells[r['source_language'], r['target_language']]
                assert r['predicted_label'] in (0, 1) and r['true_label'] in (0, 1)
                assert r['layer_index'] == selected[r['fold'], r['source_language']]
                assert key not in cell, 'Duplicate item in a cell'
                cell[key] = int(r['predicted_label'] == r['true_label'])
        keys = sorted(cells['en', 'en'])
        assert len(keys) == 5044 and len({k[0] for k in keys}) == 5044
        assert all(sorted(cell) == keys for cell in cells.values())
        if reference is None:
            reference = keys
            manifestos = sorted({k[1] for k in keys})
            assert len(manifestos) == 66
            assert all(len({k[2] for k in keys if k[1] == m}) == 1 for m in manifestos)
            cluster = {m: i for i, m in enumerate(manifestos)}
            totals = np.zeros((66, 2), dtype=np.int64)
            for _, manifesto, _, label in keys:
                totals[cluster[manifesto], label] += 1
            assert totals.sum(0).tolist() == [2496, 2548]
        assert keys == reference, 'Models are not paired on identical items and labels'
        correct = np.zeros((66, 6, 6, 2), dtype=np.int64)
        for si, source in enumerate(LANGS):
            for ti, target in enumerate(LANGS):
                for key, value in cells[source, target].items():
                    correct[cluster[key[1]], si, ti, key[3]] += value
                point = balanced(correct[:, si, ti].sum(0), totals.sum(0))
                assert abs(point - summary['matrix'][source][target]['balanced_accuracy']) < 1e-12
        counts.append(correct)
        metadata[model] = {'predictions_path': str(pred), 'predictions_sha256': hashlib.sha256(pred.read_bytes()).hexdigest(),
                           'summary_path': str(summary_path), 'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                           **{k: summary[k] for k in ['model', 'model_revision', 'selection_summary_sha256', 'input_sha256']}}
    correct = np.stack(counts, axis=1)  # manifesto, model, source, target, class
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(66, np.full(66, 1 / 66), size=args.draws)
    denominators = weights @ totals
    valid = (denominators > 0).all(axis=1)
    discarded = int((~valid).sum())
    weights = weights[valid]
    while len(weights) < args.draws:
        extra = rng.multinomial(66, np.full(66, 1 / 66), size=args.draws - len(weights))
        good = ((extra @ totals) > 0).all(axis=1)
        discarded += int((~good).sum())
        weights = np.concatenate([weights, extra[good]])
    denominators = weights @ totals
    draws = balanced((weights @ correct.reshape(66, -1)).reshape(args.draws, 4, 6, 6, 2),
                     denominators[:, None, None, None, :]) * 100
    point = balanced(correct.sum(0), totals.sum(0)) * 100
    off = ~np.eye(6, dtype=bool)
    diagonal = np.arange(6)
    result = {'method': {'date': '2026-09-30', 'metric': 'pooled out-of-fold balanced accuracy, percent',
              'unit': '66 manifestos sampled with replacement; translations and models stay paired',
              'interval': 'pointwise 95% percentile; not simultaneous or multiplicity-adjusted',
              'conditional_on': 'saved fitted probes, selected layers, and evaluation splits',
              'limitation': 'does not measure retraining uncertainty or party-independent generalization',
              'draws': args.draws, 'seed': args.seed, 'discarded_zero_class_draws': discarded,
              'items_per_cell': 5044, 'class_counts': [2496, 2548], 'point_checks': 'all 144 cells match input summaries to 1e-12',
              'fixture_checks': 'hand-computed class recall, duplicate cluster weights, identical paired difference'},
              'languages': dict(zip(LANGS, NAMES)), 'provenance': metadata, 'models': {}, 'paired_model_differences': {}}
    for mi, model in enumerate(INPUTS):
        p, d = point[mi], draws[:, mi]
        result['models'][model] = {'matrix': interval(p, d), 'off_diagonal_mean': interval(p[off].mean(), d[:, off].mean(1)),
          'diagonal_mean': interval(p[diagonal, diagonal].mean(), d[:, diagonal, diagonal].mean(1)),
          'source_means': interval((p * off).sum(1) / 5, (d * off).sum(2) / 5),
          'target_means': interval((p * off).sum(0) / 5, (d * off).sum(1) / 5),
          'source_own_minus_transfer_pp': interval(p[diagonal, diagonal][:, None] - p, d[:, diagonal, diagonal][:, :, None] - d)}
    for i, first in enumerate(INPUTS):
        for j, second in enumerate(INPUTS):
            if j > i:
                result['paired_model_differences'][f'{second} minus {first}'] = interval(
                    point[j][off].mean() - point[i][off].mean(), draws[:, j, off].mean(1) - draws[:, i, off].mean(1))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'bootstrap_results.json').write_text(json.dumps(result, indent=2) + '\n')
    def fmt(x, lo, hi):
        return f'{x:.2f} [{lo:.2f}, {hi:.2f}]'
    parts = ['<!doctype html><meta charset="utf-8"><title>RILE v2 uncertainty tables</title><style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse;margin:20px 0}th,td{padding:9px;border:1px solid #bbb}th{background:#eee}td.diag{background:#e7f1fa}</style>',
             f'<h1>RILE v2 balanced accuracy with pointwise 95% confidence intervals</h1><p>Rows: training source language. Columns: evaluation target language. Values are percent: score [lower, upper]. Blue diagonal cells use the same language. Each cell pools 5,044 held-out predictions. {args.draws:,} paired manifesto bootstrap draws; seed {args.seed}.</p>',
             '<p>Intervals condition on saved probes, selected layers, and splits. They exclude retraining uncertainty and do not provide simultaneous coverage or party-independent generalization.</p>', '<p><a href="bootstrap_results.json">Download full numerical results and provenance (JSON)</a></p>']
    for model, data in result['models'].items():
        parts.append(f'<h2>{html.escape(model)} — {html.escape(metadata[model]["model"])}</h2><table><tr><th>Source → target</th>' + ''.join(f'<th>{n}</th>' for n in NAMES) + '</tr>')
        mat = data['matrix']
        for i, name in enumerate(NAMES):
            parts.append(f'<tr><th>{name}</th>' + ''.join(f'<td class="{"diag" if i == j else ""}">{fmt(mat["point"][i][j], mat["low"][i][j], mat["high"][i][j])}</td>' for j in range(6)) + '</tr>')
        parts.append('</table>')
        for label in ['off_diagonal_mean', 'diagonal_mean']:
            parts.append(f'<p>{label.replace("_", " ")}: {fmt(**dict(zip(["x", "lo", "hi"], [data[label][k] for k in ["point", "low", "high"]])))}</p>')
        parts.append('<table><tr><th>Language</th><th>Mean as source (5 transfers)</th><th>Mean as target (5 transfers)</th></tr>')
        for i, name in enumerate(NAMES):
            parts.append(f'<tr><th>{name}</th>' + ''.join('<td>' + fmt(*[data[key][k][i] for k in ['point', 'low', 'high']]) + '</td>' for key in ['source_means', 'target_means']) + '</tr>')
        parts.append('</table>')
    parts.append('<h2>Paired differences in mean cross-language accuracy (percentage points)</h2><table>')
    for pair, values in result['paired_model_differences'].items():
        parts.append(f'<tr><th>{pair}</th><td>{fmt(*[values[k] for k in ["point", "low", "high"]])}</td></tr>')
    parts.append('</table><p>Source-own minus transfer gaps for every pair are included in the JSON. Negative gaps mean higher transfer accuracy than source-own accuracy.</p>')
    (args.output / 'tables.html').write_text('\n'.join(parts))
    print(json.dumps({'elapsed_seconds': round(time.monotonic() - started, 2), 'output': str(args.output),
                      'means': {m: r['off_diagonal_mean'] for m, r in result['models'].items()},
                      'paired_differences': result['paired_model_differences']}, indent=2))


if __name__ == '__main__':
    main()
