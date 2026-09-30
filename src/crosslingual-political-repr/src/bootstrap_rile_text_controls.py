"""Paired manifesto bootstrap of fixed text controls and saved RILE v2 probes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from bootstrap_rile_transfer import INPUTS, balanced, fixture, interval
from run_rile_text_controls import FAMILIES, LANGS


def read(path):
    cells = {(s, t): {} for s in LANGS for t in LANGS}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            key = (r['item_id'], r['manifesto_id'], r['fold'], r['true_label'])
            assert r['true_label'] in (0, 1) and r['predicted_label'] in (0, 1)
            cell = cells[r['source_language'], r['target_language']]
            assert key not in cell, 'Duplicate prediction'
            cell[key] = int(r['true_label'] == r['predicted_label'])
    keys = sorted(cells['en', 'en'])
    assert len(keys) == len({k[0] for k in keys}) == 5044
    assert all(sorted(cell) == keys for cell in cells.values()), 'Cell alignment mismatch'
    return keys, cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20260930)
    args = p.parse_args()
    assert not args.output.exists() and args.draws > 0
    fixture()
    summary = json.loads((args.controls / 'summary.json').read_text())
    assert summary['status'] == 'complete' and not summary['smoke']
    paths = {f: args.controls / f'{f}_predictions.jsonl.gz' for f in FAMILIES}
    paths.update({m: next((Path('gcp-workspace') / directory).glob('*predictions.jsonl.gz')) for m, directory in INPUTS.items()})
    reference, counts, provenance = None, [], {}
    for name, path in paths.items():
        keys, cells = read(path)
        probe_summary = json.loads(next(path.parent.glob('*summary.json')).read_text()) if name in INPUTS else None
        if reference is None:
            reference = keys
            manifestos = sorted({k[1] for k in keys})
            assert len(manifestos) == 66
            assert all(len({k[2] for k in keys if k[1] == m}) == 1 for m in manifestos)
            cluster = {m: i for i, m in enumerate(manifestos)}
            totals = np.zeros((66, 2), dtype=np.int64)
            for _, m, _, label in keys:
                totals[cluster[m], label] += 1
            assert totals.sum(0).tolist() == [2496, 2548]
        assert keys == reference, 'Controls and probes are not paired'
        correct = np.zeros((66, 6, 6, 2), dtype=np.int64)
        for si, s in enumerate(LANGS):
            for ti, t in enumerate(LANGS):
                for key, value in cells[s, t].items():
                    correct[cluster[key[1]], si, ti, key[3]] += value
                if probe_summary is not None:
                    assert abs(balanced(correct[:, si, ti].sum(0), totals.sum(0)) - probe_summary['matrix'][s][t]['balanced_accuracy']) < 1e-12
                if name in FAMILIES:
                    assert abs(balanced(correct[:, si, ti].sum(0), totals.sum(0)) - summary['matrices'][name][s][t]['balanced_accuracy']) < 1e-12
        counts.append(correct)
        provenance[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    correct = np.stack(counts, axis=1)
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(66, np.full(66, 1 / 66), size=args.draws)
    assert ((weights @ totals) > 0).all(), 'Zero-class bootstrap draw; revise explicitly'
    draws = balanced((weights @ correct.reshape(66, -1)).reshape(args.draws, 7, 6, 6, 2), (weights @ totals)[:, None, None, None, :]) * 100
    point = balanced(correct.sum(0), totals.sum(0)) * 100
    off, diagonal = ~np.eye(6, dtype=bool), np.arange(6)
    result = {'method': {'date': '2026-09-30', 'metric': 'pooled out-of-fold balanced accuracy percent', 'draws': args.draws, 'seed': args.seed, 'numpy': np.__version__, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(), 'helper_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['bootstrap_rile_transfer.py', 'run_rile_text_controls.py']}, 'unit': '66 manifestos sampled with replacement; all models and translations paired', 'interval': 'pointwise 95% percentile, not simultaneous', 'limitation': 'conditional on fixed fitted models and splits; excludes retraining uncertainty and does not establish party-independent or ideological generalization'}, 'languages': LANGS, 'provenance': provenance, 'controls': {}, 'probe_minus_control_pp': {}}
    for i, name in enumerate(FAMILIES):
        pnt, d = point[i], draws[:, i]
        result['controls'][name] = {'matrix': interval(pnt, d), 'off_diagonal_mean': interval(pnt[off].mean(), d[:, off].mean(1)), 'diagonal_mean': interval(pnt[diagonal, diagonal].mean(), d[:, diagonal, diagonal].mean(1)), 'source_means': interval((pnt * off).sum(1) / 5, (d * off).sum(2) / 5), 'target_means': interval((pnt * off).sum(0) / 5, (d * off).sum(1) / 5), 'source_own_minus_transfer_pp': interval(pnt[diagonal, diagonal][:, None] - pnt, d[:, diagonal, diagonal][:, :, None] - d)}
        for j, model in enumerate(INPUTS, start=3):
            result['probe_minus_control_pp'][f'{model} minus {name}'] = interval(point[j][off].mean() - pnt[off].mean(), draws[:, j, off].mean(1) - d[:, off].mean(1))
    args.output.mkdir(parents=True)
    (args.output / 'bootstrap_results.json').write_text(json.dumps(result, indent=2) + '\n')
    parts = ['<!doctype html><meta charset="utf-8"><title>RILE v2 text controls</title><style>body{font:16px system-ui;margin:32px}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>RILE v2 text controls</h1><p>Balanced accuracy percent, pointwise 95% intervals. Rows are source languages; columns are target languages. Each cell pools 5,044 predictions. Fixed-model, paired manifesto bootstrap; 10,000 draws by default. No retraining uncertainty.</p><a href="bootstrap_results.json">Full numerical results and provenance</a>']
    for name, data in result['controls'].items():
        mat = data['matrix']
        parts.append(f'<h2>{name}</h2><table><tr><th>Source / target</th>' + ''.join(f'<th>{t}</th>' for t in LANGS) + '</tr>')
        for i, s in enumerate(LANGS):
            parts.append(f'<tr><th>{s}</th>' + ''.join(f'<td>{mat["point"][i][j]:.2f} [{mat["low"][i][j]:.2f}, {mat["high"][i][j]:.2f}]</td>' for j in range(6)) + '</tr>')
        parts.append('</table>')
    (args.output / 'tables.html').write_text('\n'.join(parts))
    print(json.dumps({'status': 'complete', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
