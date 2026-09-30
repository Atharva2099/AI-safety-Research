"""Frozen category holdout: each item uses only its manifesto-excluding fold."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import platform
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
from bootstrap_rile_transfer import INPUTS, LANGS

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'src/crosslingual-political-repr/data/rile_v2/final_translations.jsonl'
CONTROLS = ROOT / 'gcp-workspace/rile_v2_text_controls/full-20260930-luna'
PINS = {
    'OLMo': ('allenai/Olmo-3-7B-Instruct', '6e5971d9eba42665f5bd5a0fcf047f299ce1dccc'),
    'Qwen3.5': ('Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
    'Ministral': ('mistralai/Ministral-8B-Instruct-2410', '2f494a194c5b980dfb9772cb92d26cbb671fce5a'),
    'Gemma': ('google/gemma-2-9b-it', '11c9b309abf73637e4b6f9a3fa1e92e615547819'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(model, suffix):
    paths = list((ROOT / 'gcp-workspace' / INPUTS[model]).glob(f'*{suffix}'))
    assert len(paths) == 1, f'Ambiguous {model} {suffix}'
    return paths[0]


def records(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def load_cohorts():
    assert sha(DATA) == 'fe6eb10383707b44a0fdfbbf65b95f6258c0ef459c30fec8240e8449f01700ca'
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len(rows) == len({r['item_id'] for r in rows}) == 6131
    assert all(type(r['held_out']) is bool and r['label'] in (0, 1) and
               all(isinstance(r[l], str) and r[l] for l in LANGS) for r in rows)
    dev, hold = ([r for r in rows if not r['held_out']], [r for r in rows if r['held_out']])
    by_id = {r['item_id']: r for r in dev}
    foldmap, seen = {}, set()
    for r in records(artifact('OLMo', 'predictions.jsonl.gz')):
        if (r['source_language'], r['target_language']) != ('en', 'en'):
            continue
        item = by_id[r['item_id']]
        assert r['item_id'] not in seen and (item['manifesto_id'], item['label']) == (r['manifesto_id'], r['true_label'])
        seen.add(r['item_id'])
        assert r['fold'] in range(5)
        assert foldmap.setdefault(r['manifesto_id'], r['fold']) == r['fold'], 'Manifesto spans folds'
    assert seen == set(by_id) and len(foldmap) == 66
    dev.sort(key=lambda r: r['item_id'])  # Exact text-control training row order.
    hold.sort(key=lambda r: r['item_id'])
    assert len(dev) == 5044 and Counter(r['label'] for r in dev) == {0: 2496, 1: 2548}
    assert len(hold) == 1087 and Counter(r['label'] for r in hold) == {0: 593, 1: 494}
    assert Counter(r['code'] for r in hold) == {'107': 287, '506': 306, '601': 271, '603': 223}
    assert len({r['manifesto_id'] for r in hold}) == 58
    assert all(r['manifesto_id'] in foldmap for r in hold)
    assert np.bincount([foldmap[r['manifesto_id']] for r in hold]).tolist() == [165, 167, 178, 251, 326]
    return dev, hold, foldmap


def sample(rows, foldmap):
    # Fixed metadata selection: two item IDs from every present fold/label stratum.
    return [r for f in range(5) for y in (0, 1)
            for r in [r for r in rows if foldmap[r['manifesto_id']] == f and r['label'] == y][:2]]


def matrix(rows):
    result = {}
    for s in sorted({r['source_language'] for r in rows}, key=LANGS.index):
        result[s] = {}
        for t in LANGS:
            cell = [r for r in rows if (r['source_language'], r['target_language']) == (s, t)]
            totals = np.bincount([r['true_label'] for r in cell], minlength=2)
            correct = np.bincount([r['true_label'] for r in cell if r['true_label'] == r['predicted_label']], minlength=2)
            assert (totals > 0).all()
            result[s][t] = {'items': len(cell), 'class_counts': totals.tolist(),
                            'balanced_accuracy': float((correct / totals).mean())}
    return result


def prediction(item, fold, source, target, guess, score, **extra):
    assert guess in (0, 1) and np.isfinite(score)
    return {**{k: item.get(k) for k in ['item_id', 'manifesto_id', 'code', 'cmp_code', 'source_lang']},
            'fold': int(fold), 'source_language': source, 'target_language': target,
            'true_label': int(item['label']), 'predicted_label': int(guess), 'decision_score': float(score), **extra}


def write_predictions(output, name, rows):
    path = output / f'{name}_predictions.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    return {'path': str(path), 'sha256': sha(path), 'rows': len(rows)}


def model_run(name, dev, hold, foldmap, smoke, output):
    import torch
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from multilingual_layerwise_probe import choose_device, extract_rile_memmap, replay_snapshot_revision, model_dimensions
    summary_path, probe_path = artifact(name, 'summary.json'), artifact(name, 'probes.npz')
    prior = json.loads(summary_path.read_text())
    model_id, revision = PINS[name]
    assert prior['status'] == 'complete' and (prior['model'], prior['model_revision']) == (model_id, revision)
    assert prior['input_sha256'] == sha(DATA)
    layers = {(r['fold'], r['source_language']): r['layer_index'] for r in prior['selected_layers']}
    assert set(layers) == {(f, s) for f in range(5) for s in LANGS}
    params = np.load(probe_path)
    assert set(params.files) == {f'f{f}_{s}_{k}' for f in range(5) for s in LANGS for k in ('coef', 'intercept', 'classes')}
    cached = try_to_load_from_cache(model_id, 'config.json', revision=revision)
    assert isinstance(cached, str) and Path(cached).is_file(), 'Pinned model absent from local cache'
    snapshot = Path(cached).parent
    replay_snapshot_revision(snapshot, revision, None)
    device = choose_device('cuda')
    assert torch.cuda.is_available(), 'Model extraction requires GPU scratch, not a local activation cache'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
        **({'fix_mistral_regex': True} if name == 'Ministral' else {}))
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=dtype).to(device).eval()
    replay_snapshot_revision(snapshot, revision, getattr(model.config, '_commit_hash', None))
    assert all(0 <= l < model_dimensions(model.config)[0] for l in layers.values())
    # First eight canonical rows retain the original two batch-of-four contexts.
    parity_rows = [r for r in map(json.loads, DATA.read_text().splitlines()) if not r['held_out']][:8]
    parity_ids = {r['item_id'] for r in parity_rows}
    expected = {}
    for r in records(artifact(name, 'predictions.jsonl.gz')):
        assert r['fold'] == foldmap[r['manifesto_id']], 'Saved model fold differs from document reference'
        if r['item_id'] in parity_ids:
            key = (r['item_id'], r['source_language'], r['target_language'])
            assert key not in expected
            expected[key] = r
    assert len(expected) == len(parity_rows) * 36
    predictions, parity_errors = [], []
    with tempfile.TemporaryDirectory(prefix='rile-category-holdout-') as scratch:
        for target in LANGS:
            combined = parity_rows + hold
            items = [{'statement': r[target]} for r in combined]
            vectors = extract_rile_memmap(model, tokenizer, items, device, Path(scratch) / f'{target}.f16', 4)
            for f in range(5):
                indices = [i for i, r in enumerate(combined) if foldmap[r['manifesto_id']] == f]
                for source in LANGS:
                    key, layer = f'f{f}_{source}', layers[f, source]
                    coef, intercept, classes = (params[f'{key}_{k}'] for k in ('coef', 'intercept', 'classes'))
                    assert classes.tolist() == [0, 1] and coef.shape == (1, vectors.shape[2]) and intercept.shape == (1,)
                    scores = (np.asarray(vectors[indices, layer, :], dtype=np.float32) @ coef.T + intercept).ravel()
                    guesses = classes[(scores > 0).astype(int)]
                    for i, guess, score in zip(indices, guesses, scores):
                        r = combined[i]
                        if i < len(parity_rows):
                            old = expected[r['item_id'], source, target]
                            error = abs(float(score) - old['decision_score'])
                            parity_errors.append(error)
                            # Same precision, batches and cached snapshot: only small GPU numerical drift is allowed.
                            assert int(guess) == old['predicted_label'] and np.isclose(score, old['decision_score'], atol=1e-3, rtol=1e-4), f'DEV extraction parity failed: {name} {r["item_id"]} {source} {target} error={error}'
                        else:
                            predictions.append(prediction(r, f, source, target, guess, score, layer_index=layer))
            del vectors
            (Path(scratch) / f'{target}.f16').unlink()
    assert len(predictions) == len(hold) * 36
    return {name: predictions}, {'selected_layers': prior['selected_layers'], 'model': model_id, 'model_revision': revision,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'device': str(device), 'dtype': str(dtype),
        'feature': 'raw final-token hidden states stored float16 then cast float32; batch size 4',
        'dev_parity': {'items': len(parity_rows), 'checks': len(parity_errors), 'max_absolute_margin_error': max(parity_errors), 'atol': .001, 'rtol': .0001, 'guesses': 'exact'},
        'saved_summary': {'path': str(summary_path), 'sha256': sha(summary_path)}, 'saved_parameters': {'path': str(probe_path), 'sha256': sha(probe_path)}}


def control_run(dev, hold, foldmap, smoke):
    import jieba
    import sklearn
    import run_rile_text_controls as c
    assert (np.__version__, sklearn.__version__, jieba.__version__, platform.python_version()) == ('2.5.3', '1.9.1', '0.42.1', '3.12.7'), 'Use the frozen CPU dependency versions'
    prior = json.loads((CONTROLS / 'summary.json').read_text())
    assert prior['status'] == 'complete' and prior['provenance']['input_sha256'] == sha(DATA)
    segmenter = jieba.Tokenizer(); segmenter.tmp_dir = '/tmp'; segmenter.cache_file = 'rile-text-controls-jieba.cache'; segmenter.initialize()
    c.fixture(segmenter)
    assert c.sha(Path(jieba.__file__).parent / 'dict.txt') == prior['provenance']['jieba_dictionary_sha256']
    assert c.unicodedata.unidata_version == prior['provenance']['unicode']
    y = np.array([r['label'] for r in dev]); folds = np.array([foldmap[r['manifesto_id']] for r in dev])
    texts = {l: [r[l] for r in dev + hold] for l in LANGS}
    words = {l: [' '.join(c.word_tokens(t, l, segmenter)) for t in texts[l]] for l in LANGS}
    numeric = {l: np.array([c.surface(t) for t in texts[l]], dtype=float) for l in LANGS}
    fitted = {(r['family'], r['source'], r['fold']): r for r in prior['fits']}
    predictions, fits = {}, []
    for family in c.FAMILIES:
        predictions[family] = []
        for source in (['en'] if smoke else LANGS):
            for f in ([0] if smoke else range(5)):
                train = folds != f
                indices = [i for i, r in enumerate(hold) if foldmap[r['manifesto_id']] == f]
                assert not ({r['manifesto_id'] for i, r in enumerate(dev) if train[i]} & {hold[i]['manifesto_id'] for i in indices})
                if family == 'surface':
                    v = c.StandardScaler().fit(numeric[source][:len(dev)][train]); x = v.transform(numeric[source][:len(dev)][train])
                else:
                    values = texts if family == 'char' else words
                    v = c.TfidfVectorizer(analyzer='char', ngram_range=(3, 5)) if family == 'char' else c.TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2))
                    x = v.fit_transform(np.array(values[source][:len(dev)])[train])
                clf = c.LogisticRegression(C=1, max_iter=3000, solver='lbfgs', random_state=42)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always'); clf.fit(x, y[train])
                assert not caught, f'Fit warning needs review: {[str(w.message) for w in caught]}'
                hashes = {'coef_sha256': c.array_sha(clf.coef_), 'intercept_sha256': c.array_sha(clf.intercept_)}
                hashes.update({'scaler_mean_sha256': c.array_sha(v.mean_), 'scaler_scale_sha256': c.array_sha(v.scale_)} if family == 'surface' else {'vocabulary_sha256': hashlib.sha256(json.dumps(v.vocabulary_, sort_keys=True).encode()).hexdigest(), 'idf_sha256': c.array_sha(v.idf_)})
                old = fitted[family, source, f]
                assert all(old[k] == value for k, value in hashes.items()), f'Frozen control hash parity failed: {family} {source} fold {f}'
                fit = {'family': family, 'source': source, 'fold': f, 'train_items': int(train.sum()), 'test_items': len(indices), 'warnings': [], **hashes, 'targets': {}}
                for target in LANGS:
                    values_test = numeric[target][len(dev):][indices] if family == 'surface' else np.array(values[target][len(dev):])[indices]
                    xtest = v.transform(values_test); guesses, scores = clf.predict(xtest), clf.decision_function(xtest)
                    if family != 'surface':
                        nnz = np.asarray(xtest.getnnz(axis=1)).ravel()
                        fit['targets'][target] = {'empty_rows': int((nnz == 0).sum()), 'empty_fraction': float((nnz == 0).mean()), 'mean_matched_features': float(nnz.mean())}
                    predictions[family].extend(prediction(hold[i], f, source, target, guess, score) for i, guess, score in zip(indices, guesses, scores))
                fits.append(fit)
                print(f'{family} {source} fold={f}: exact frozen-fit hashes matched', flush=True)
    return predictions, {'fits': fits, 'frozen_summary': {'path': str(CONTROLS / 'summary.json'), 'sha256': sha(CONTROLS / 'summary.json')}, 'sklearn': sklearn.__version__, 'jieba': jieba.__version__}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=[*PINS, 'controls'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); started = time.monotonic()
    assert not args.output.exists(), 'Output exists; use a separate smoke or full output'
    dev, hold, foldmap = load_cohorts()
    if args.smoke:
        hold = sample(hold, foldmap)
        if args.model == 'controls':
            hold = [r for r in hold if foldmap[r['manifesto_id']] == 0]
    args.output.mkdir(parents=True)
    summary = {'status': 'needs_review', 'date': '2026-09-30', 'smoke': args.smoke, 'protocol': 'frozen_category_holdout', 'input': {'path': str(DATA), 'sha256': sha(DATA)}, 'fold_reference': {'path': str(artifact('OLMo', 'predictions.jsonl.gz')), 'sha256': sha(artifact('OLMo', 'predictions.jsonl.gz'))}, 'foldmap': foldmap, 'cohort_items': len(hold), 'cohort_class_counts': dict(Counter(r['label'] for r in hold)), 'cohort_manifestos': len({r['manifesto_id'] for r in hold}), 'cohort_fold_counts': dict(Counter(foldmap[r['manifesto_id']] for r in hold)), 'python': platform.python_version(), 'numpy': np.__version__, 'code_sha256': sha(__file__)}
    helper_names = ['bootstrap_rile_transfer.py', 'run_rile_text_controls.py' if args.model == 'controls' else 'multilingual_layerwise_probe.py']
    summary['helper_sha256'] = {name: sha(Path(__file__).with_name(name)) for name in helper_names}
    try:
        predictions, metadata = control_run(dev, hold, foldmap, args.smoke) if args.model == 'controls' else model_run(args.model, dev, hold, foldmap, args.smoke, args.output)
        summary.update(metadata)
        summary['prediction_files'] = {name: write_predictions(args.output, name, rows) for name, rows in predictions.items()}
        summary['matrices'] = {name: matrix(rows) for name, rows in predictions.items()}
        summary['status'] = 'smoke_complete' if args.smoke else 'complete'
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'output': str(args.output), 'elapsed_seconds': summary['elapsed_seconds']}))


if __name__ == '__main__':
    main()
