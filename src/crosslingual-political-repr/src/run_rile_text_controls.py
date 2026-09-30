"""Fixed CPU text controls on the saved RILE v2 manifesto folds."""
import argparse
import gzip
import hashlib
import json
import platform
import time
import unicodedata
import warnings
from pathlib import Path

import jieba
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

LANGS = ['en', 'es', 'de', 'zh', 'hi', 'mr']
FAMILIES = ['char', 'word', 'surface']
DATA = Path('src/crosslingual-political-repr/data/rile_v2/final_translations.jsonl')
REFERENCE = Path('gcp-workspace/rile-v2-source-matrix-replay-20260929-retrieval/olmo/rile_v2_source_matrix_replay_allenai_Olmo-3-7B-Instruct_predictions.jsonl.gz')
DATA_SHA = 'fe6eb10383707b44a0fdfbbf65b95f6258c0ef459c30fec8240e8449f01700ca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def load():
    assert sha(DATA) == DATA_SHA, 'Dataset changed'
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    rows = [r for r in rows if not r['held_out']]
    by_id = {r['item_id']: r for r in rows}
    assert len(by_id) == len(rows) == 5044
    folds = {}
    with gzip.open(REFERENCE, 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['source_language'] != 'en' or r['target_language'] != 'en':
                continue
            item = by_id[r['item_id']]
            assert (item['manifesto_id'], item['label']) == (r['manifesto_id'], r['true_label'])
            assert r['item_id'] not in folds, 'Duplicate fold reference'
            folds[r['item_id']] = r['fold']
    assert set(folds) == set(by_id)
    rows.sort(key=lambda r: r['item_id'])
    fold = np.array([folds[r['item_id']] for r in rows])
    assert np.bincount(fold).tolist() == [1011, 1007, 1009, 1007, 1010]
    groups = {r['manifesto_id'] for r in rows}
    assert len(groups) == 66
    assert all(len({fold[i] for i, r in enumerate(rows) if r['manifesto_id'] == m}) == 1 for m in groups)
    assert [len({r['manifesto_id'] for i, r in enumerate(rows) if fold[i] == f}) for f in range(5)] == [12, 14, 13, 14, 13]
    assert np.bincount([r['label'] for r in rows]).tolist() == [2496, 2548]
    return rows, fold


def word_tokens(text, language, segmenter):
    text = text.lower()
    if language == 'zh':
        return [t for t in segmenter.cut(text, HMM=False) if any(unicodedata.category(c)[0] in 'LMN' for c in t)]
    tokens, run = [], ''
    for c in text:
        if unicodedata.category(c)[0] in 'LMN':
            run += c
        elif run:
            tokens.append(run)
            run = ''
    return tokens + ([run] if run else [])


def surface(text):
    stripped = text.rstrip()
    n, segments = len(text), len(text.split())
    last = stripped[-1:] or ''
    return [n, segments, segments / max(n, 1)] + [int(last in chars) if last else 0 for chars in ['.。', '?？', '!！', ':：', ';；', ',，', '।॥']] + [int(bool(last) and unicodedata.category(last).startswith('P'))]


def fixture(segmenter):
    assert word_tokens('भारत की नीति', 'hi', segmenter) == ['भारत', 'की', 'नीति']
    assert len(word_tokens('中国政策', 'zh', segmenter)) >= 2
    v = TfidfVectorizer(tokenizer=str.split, token_pattern=None)
    v.fit(['training shared', 'other shared'])
    assert 'heldout' not in v.vocabulary_ and v.transform(['heldout']).nnz == 0
    assert surface('x?')[-1] == 1 and surface('')[-1] == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='Only English source, fold 0; all families and targets')
    args = p.parse_args()
    assert not args.output.exists(), 'Output already exists; use a distinct path'
    started = time.monotonic()
    segmenter = jieba.Tokenizer()
    segmenter.tmp_dir = '/tmp'
    segmenter.cache_file = 'rile-text-controls-jieba.cache'
    segmenter.initialize()
    fixture(segmenter)
    rows, folds = load()
    y = np.array([r['label'] for r in rows])
    texts = {lang: [r[lang] for r in rows] for lang in LANGS}
    words = {lang: [' '.join(word_tokens(t, lang, segmenter)) for t in texts[lang]] for lang in LANGS}
    numeric = {lang: np.array([surface(t) for t in texts[lang]], dtype=float) for lang in LANGS}
    dictionary = Path(jieba.__file__).parent / 'dict.txt'
    summary = {'status': 'running', 'date': '2026-09-30', 'smoke': args.smoke,
        'provenance': {'input_path': str(DATA), 'input_sha256': sha(DATA), 'fold_reference': str(REFERENCE), 'fold_reference_sha256': sha(REFERENCE), 'code_sha256': sha(__file__),
            'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'jieba': jieba.__version__, 'unicode': unicodedata.unidata_version, 'jieba_dictionary_sha256': sha(dictionary)},
        'settings': {'char': 'char 3-5, lowercase, raw tf, smooth idf, l2 norm', 'word': 'word 1-2, Unicode LMN runs; Chinese jieba HMM=False; lowercase; raw tf, smooth idf, l2 norm', 'surface': 'character length, whitespace segments, segments/characters, seven final punctuation flags and Unicode final punctuation flag; training-only StandardScaler',
            'classifier': {'C': 1, 'max_iter': 3000, 'solver': 'lbfgs', 'random_state': 42}, 'tuning': 'none', 'folds': 'saved reference; no resplitting'}, 'fits': [], 'matrices': {}}
    args.output.mkdir(parents=True)
    any_warning = False
    for family in FAMILIES:
        cells = {(s, t): [[], []] for s in (['en'] if args.smoke else LANGS) for t in LANGS}
        path = args.output / f'{family}_predictions.jsonl.gz'
        with gzip.open(path, 'wt') as stream:
            for source in (['en'] if args.smoke else LANGS):
                for fold in ([0] if args.smoke else range(5)):
                    train, test = folds != fold, folds == fold
                    if family == 'surface':
                        vectorizer = StandardScaler().fit(numeric[source][train])
                        xtrain = vectorizer.transform(numeric[source][train])
                    else:
                        vectorizer = (TfidfVectorizer(analyzer='char', ngram_range=(3, 5)) if family == 'char' else TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2)))
                        values = texts if family == 'char' else words
                        xtrain = vectorizer.fit_transform(np.array(values[source])[train])
                    clf = LogisticRegression(C=1, max_iter=3000, solver='lbfgs', random_state=42)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always', ConvergenceWarning)
                        clf.fit(xtrain, y[train])
                    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                    any_warning |= bool(messages)
                    fit = {'family': family, 'source': source, 'fold': int(fold), 'train_items': int(train.sum()), 'test_items': int(test.sum()), 'features': xtrain.shape[1], 'n_iter': clf.n_iter_.tolist(), 'warnings': messages, 'coef_sha256': array_sha(clf.coef_), 'intercept_sha256': array_sha(clf.intercept_), 'targets': {}}
                    if family != 'surface':
                        fit.update(vocabulary_sha256=hashlib.sha256(json.dumps(vectorizer.vocabulary_, sort_keys=True).encode()).hexdigest(), idf_sha256=array_sha(vectorizer.idf_))
                    else:
                        fit.update(scaler_mean_sha256=array_sha(vectorizer.mean_), scaler_scale_sha256=array_sha(vectorizer.scale_))
                    for target in LANGS:
                        xtest = vectorizer.transform(numeric[target][test] if family == 'surface' else np.array(values[target])[test])
                        prediction, scores = clf.predict(xtest), clf.decision_function(xtest)
                        if family != 'surface':
                            nonzero = np.asarray(xtest.getnnz(axis=1)).ravel()
                            fit['targets'][target] = {'empty_rows': int((nonzero == 0).sum()), 'empty_fraction': float((nonzero == 0).mean()), 'mean_matched_features': float(nonzero.mean())}
                        cells[source, target][0].extend(y[test].tolist())
                        cells[source, target][1].extend(prediction.tolist())
                        for i, predicted, score in zip(np.flatnonzero(test), prediction, scores):
                            r = rows[i]
                            stream.write(json.dumps({'source_language': source, 'target_language': target, 'fold': int(fold), 'item_id': r['item_id'], 'manifesto_id': r['manifesto_id'], 'true_label': int(y[i]), 'predicted_label': int(predicted), 'decision_score': float(score)}, ensure_ascii=False) + '\n')
                    summary['fits'].append(fit)
                    print(f'{family} source={source} fold={fold} iterations={clf.n_iter_.tolist()} warnings={len(messages)}', flush=True)
        summary['matrices'][family] = {s: {t: {'n': len(cells[s, t][0]), 'balanced_accuracy': balanced_accuracy_score(*cells[s, t]), 'accuracy': accuracy_score(*cells[s, t]), 'f1_label_1': f1_score(*cells[s, t]), 'class_counts': np.bincount(cells[s, t][0], minlength=2).tolist()} for t in LANGS} for s in (['en'] if args.smoke else LANGS)}
        summary.setdefault('prediction_files', {})[family] = {'path': str(path), 'sha256': sha(path)}
    summary.update(status='needs_review' if any_warning else ('smoke_complete' if args.smoke else 'complete'), elapsed_seconds=time.monotonic() - started)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    assert not any_warning, 'Convergence warning; outputs retained with needs_review status'
    print(json.dumps({'status': summary['status'], 'elapsed_seconds': summary['elapsed_seconds'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
