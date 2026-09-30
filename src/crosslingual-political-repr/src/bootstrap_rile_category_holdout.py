"""Paired manifesto bootstrap for seven frozen category-holdout families."""
import argparse
import hashlib
import html
import json
import platform
from pathlib import Path
import time

import numpy as np
from bootstrap_rile_transfer import LANGS, NAMES, balanced, interval, fixture
from evaluate_rile_category_holdout import artifact, records, sha, load_cohorts, CONTROLS, PINS


def read_cells(path, expected, layers=None):
    cells = {(s, t): {} for s in LANGS for t in LANGS}
    for r in records(path):
        cell = cells[r['source_language'], r['target_language']]
        key = r['item_id']
        assert key not in cell, 'Duplicate prediction item in cell'
        assert key in expected and (r['manifesto_id'], r['fold'], r['true_label']) == expected[key], 'Cohort label/fold alignment failed'
        assert r['predicted_label'] in (0, 1) and np.isfinite(r['decision_score'])
        if layers is not None:
            assert r['layer_index'] == layers[r['fold'], r['source_language']]
        cell[key] = int(r['true_label'] == r['predicted_label'])
    assert all(set(c) == set(expected) for c in cells.values()), 'Missing or extra item in a cell'
    return cells


def counts(cells, rows, cluster):
    result = np.zeros((len(cluster), 6, 6, 2), dtype=np.int64)
    for si, s in enumerate(LANGS):
        for ti, t in enumerate(LANGS):
            for r in rows:
                result[cluster[r['manifesto_id']], si, ti, r['label']] += cells[s, t][r['item_id']]
    return result


def totals(rows, cluster):
    out = np.zeros((len(cluster), 2), dtype=np.int64)
    for r in rows:
        out[cluster[r['manifesto_id']], r['label']] += 1
    return out


def aggregates(p, d):
    off = ~np.eye(6, dtype=bool); diag = np.arange(6)
    return {'matrix': interval(p, d), 'all_cell_mean': interval(p.mean(), d.mean((1, 2))),
        'off_diagonal_mean': interval(p[off].mean(), d[:, off].mean(1)),
        'diagonal_mean': interval(p[diag, diag].mean(), d[:, diag, diag].mean(1)),
        'source_means': interval((p * off).sum(1) / 5, (d * off).sum(2) / 5),
        'target_means': interval((p * off).sum(0) / 5, (d * off).sum(1) / 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, required=True, help='Parent with OLMo, Qwen3.5, Ministral, Gemma, controls folders')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args(); started = time.monotonic()
    assert args.draws > 0 and not args.output.exists()
    fixture()
    dev, hold, folds = load_cohorts()
    manifestos = sorted({r['manifesto_id'] for r in hold}); cluster = {m: i for i, m in enumerate(manifestos)}
    matched_dev = [r for r in dev if r['manifesto_id'] in cluster]
    expected_hold = {r['item_id']: (r['manifesto_id'], folds[r['manifesto_id']], r['label']) for r in hold}
    expected_dev = {r['item_id']: (r['manifesto_id'], folds[r['manifesto_id']], r['label']) for r in dev}
    ht, dt = totals(hold, cluster), totals(matched_dev, cluster)
    assert len(cluster) == 58 and not (set(expected_hold) & set(expected_dev))
    hold_counts, dev_counts, provenance, original = [], [], {}, {}
    families = [*PINS, 'char', 'word', 'surface']
    for family in families:
        folder = args.holdout / (family if family in PINS else 'controls')
        sp = folder / 'summary.json'; summary = json.loads(sp.read_text())
        assert summary['status'] == 'complete' and not summary['smoke'] and summary['foldmap'] == folds
        assert summary['input']['sha256'] == sha(__import__('evaluate_rile_category_holdout').DATA)
        hp = folder / f'{family}_predictions.jsonl.gz'
        assert sha(hp) == summary['prediction_files'][family]['sha256']
        if family in PINS:
            dp, dsp = artifact(family, 'predictions.jsonl.gz'), artifact(family, 'summary.json')
            ds = json.loads(dsp.read_text()); dm = ds['matrix']
            assert ds['status'] == 'complete' and (ds['model'], ds['model_revision']) == PINS[family]
            layers = {(r['fold'], r['source_language']): r['layer_index'] for r in ds['selected_layers']}
            assert summary['selected_layers'] == ds['selected_layers']
            assert summary['saved_parameters']['sha256'] == sha(artifact(family, 'probes.npz'))
        else:
            dp, dsp = CONTROLS / f'{family}_predictions.jsonl.gz', CONTROLS / 'summary.json'
            ds = json.loads(dsp.read_text()); dm = ds['matrices'][family]; layers = None
            assert ds['status'] == 'complete' and sha(dp) == ds['prediction_files'][family]['sha256']
        hc = read_cells(hp, expected_hold, layers); dc = read_cells(dp, expected_dev, layers)
        hcount, dcount = counts(hc, hold, cluster), counts(dc, matched_dev, cluster)
        for i, s in enumerate(LANGS):
            for j, t in enumerate(LANGS):
                hpnt = balanced(hcount[:, i, j].sum(0), ht.sum(0))
                dpnt = np.mean([np.mean([dc[s, t][r['item_id']] for r in dev if r['label'] == y]) for y in (0, 1)])
                assert abs(hpnt - summary['matrices'][family][s][t]['balanced_accuracy']) < 1e-12
                assert abs(dpnt - dm[s][t]['balanced_accuracy']) < 1e-12
        hold_counts.append(hcount); dev_counts.append(dcount)
        original[family] = dm
        provenance[family] = {k: {'path': str(p), 'sha256': sha(p)} for k, p in [('holdout_predictions', hp), ('holdout_summary', sp), ('development_predictions', dp), ('development_summary', dsp)]}
    hcorrect, dcorrect = np.stack(hold_counts, axis=1), np.stack(dev_counts, axis=1)
    rng = np.random.default_rng(args.seed); weights = np.empty((0, 58), dtype=np.int64); discarded = 0
    while len(weights) < args.draws:
        new = rng.multinomial(58, np.full(58, 1 / 58), size=args.draws - len(weights))
        good = ((new @ ht) > 0).all(1) & ((new @ dt) > 0).all(1)
        discarded += int((~good).sum()); weights = np.concatenate([weights, new[good]])
    def scores(correct, total):
        point = balanced(correct.sum(0), total.sum(0)) * 100
        draw = balanced((weights @ correct.reshape(58, -1)).reshape(args.draws, 7, 6, 6, 2), (weights @ total)[:, None, None, None, :]) * 100
        return point, draw
    hp, hd = scores(hcorrect, ht); dp, dd = scores(dcorrect, dt)
    result = {'method': {'date': '2026-09-30', 'metric': 'pooled balanced accuracy percent, then equal cell means', 'cluster': '58 manifestos, shared weights across families, languages and cohorts', 'draws': args.draws, 'seed': args.seed, 'discarded_zero_class_draws': discarded, 'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(), 'interval': 'pointwise 95% percentile; not simultaneous or multiplicity adjusted', 'conditional_on': 'saved probes/layers, exact reproduced controls and saved folds', 'holdout_items': len(hold), 'holdout_class_counts': ht.sum(0).tolist(), 'matched_development_items': len(matched_dev), 'matched_development_class_counts': dt.sum(0).tolist(), 'manifestos': manifestos, 'foldmap': folds, 'difference': 'holdout minus development restricted to these same 58 documents; different items/categories, descriptive category shift', 'limitations': 'No retraining uncertainty, party independence, sentence-level ideology or category-population inference from four fixed categories.', 'document_exclusion': 'Each prediction uses only the fold classifier whose training excludes its manifesto, even though both cohorts reuse these documents.', 'point_checks': '252 holdout and 252 original-development cells match summaries within 1e-12'}, 'provenance': provenance, 'code_sha256': sha(__file__), 'languages': dict(zip(LANGS, NAMES)), 'families': {}, 'probe_minus_control': {}, 'full_66_manifesto_development_context_only': original}
    result['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__}
    result['helper_sha256'] = {name: sha(Path(__file__).with_name(name)) for name in ['bootstrap_rile_transfer.py', 'evaluate_rile_category_holdout.py']}
    for i, family in enumerate(families):
        result['families'][family] = {'holdout': aggregates(hp[i], hd[:, i]), 'matched_development': aggregates(dp[i], dd[:, i]), 'holdout_minus_matched_development_pp': aggregates(hp[i] - dp[i], hd[:, i] - dd[:, i])}
    for i, probe in enumerate(PINS):
        for j, control in enumerate(families[4:], 4):
            result['probe_minus_control'][f'{probe} minus {control}'] = aggregates(hp[i] - hp[j], hd[:, i] - hd[:, j])
    args.output.mkdir(parents=True)
    result['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'bootstrap_results.json').write_text(json.dumps(result, indent=2) + '\n')
    parts = ['<!doctype html><meta charset="utf-8"><title>Frozen category holdout</title><style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #bbb}th{background:#eee}</style><h1>Frozen category holdout</h1>', f'<p>Rows are source languages; columns are target languages. Each cell pools 1,087 items. Balanced accuracy (%) [pointwise 95% interval], {args.draws:,} paired bootstrap draws of 58 manifestos. Each prediction excludes its document from training.</p><p>Conditional on saved classifiers and folds; no party independence, sentence ideology or category-population claim. <a href="bootstrap_results.json">Full matrices, paired differences, matched-document development comparison and provenance</a>.</p>']
    for family, values in result['families'].items():
        mat = values['holdout']['matrix']; parts.append(f'<h2>{html.escape(family)}</h2><table><tr><th>Source → target</th>' + ''.join(f'<th>{n}</th>' for n in NAMES) + '</tr>')
        for i, name in enumerate(NAMES):
            parts.append(f'<tr><th>{name}</th>' + ''.join(f'<td>{mat["point"][i][j]:.2f} [{mat["low"][i][j]:.2f}, {mat["high"][i][j]:.2f}]</td>' for j in range(6)) + '</tr>')
        parts.append('</table>')
    (args.output / 'tables.html').write_text('\n'.join(parts))
    print(json.dumps({'output': str(args.output), 'elapsed_seconds': result['elapsed_seconds'], 'matched_development_items': len(matched_dev)}))


if __name__ == '__main__':
    main()
