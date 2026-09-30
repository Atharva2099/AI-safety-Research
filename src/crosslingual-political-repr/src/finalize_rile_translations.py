"""Assemble the owner-approved RILE translations and decisions into one local dataset.

    uv run python -m src.finalize_rile_translations

The output contains manifesto text and is gitignored. Source files remain unchanged.
"""
import json
import argparse
import collections
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rile_v1"
LANGUAGES = ("en", "es", "de", "zh", "hi", "mr")
EXCLUDED_SUBCODES = {"202.2", "605.2", "703.2"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_v2():
    source = DATA / "final_translations.jsonl"
    rows = [r for r in read_jsonl(source) if str(r["cmp_code"]) not in EXCLUDED_SUBCODES]
    parent = {r["item_id"]: r["item_id"] for r in rows}

    def find(item_id):
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    for lang in LANGUAGES:
        seen = {}
        for row in rows:
            key = unicodedata.normalize("NFKC", row[lang]).strip().casefold()
            if key in seen:
                parent[find(row["item_id"])] = find(seen[key])
            else:
                seen[key] = row["item_id"]

    groups = collections.defaultdict(list)
    for row in rows:
        groups[find(row["item_id"])].append(row)

    retained = []
    conflicts = redundant = 0
    for group in groups.values():
        if len(group) == 1:
            retained.extend(group)
        elif len({(r["label"], r["code"], r["held_out"]) for r in group}) > 1:
            conflicts += len(group)
        else:
            retained.append(min(group, key=lambda r: r["item_id"]))
            redundant += len(group) - 1
    retained.sort(key=lambda r: r["item_id"])

    out = ROOT / "data" / "rile_v2"
    out.mkdir(exist_ok=True)
    meta_fields = ("item_id", "source_lang", "label", "cmp_code", "code", "pair_issue", "held_out",
                   "manifesto_id", "party", "partyname", "date")
    (out / "items.jsonl").write_text("".join(
        json.dumps({k: r[k] for k in meta_fields}, ensure_ascii=False) + "\n" for r in retained), encoding="utf-8")
    (out / "final_translations.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in retained), encoding="utf-8")
    counts = lambda key: dict(sorted(collections.Counter(r[key] for r in retained).items()))
    manifest = {
        "version": "rile_v2", "built": "2026-09-27", "status": "corrected dataset candidate; not evaluated",
        "source": "data/rile_v1/final_translations.jsonl", "languages": list(LANGUAGES),
        "excluded_subcodes": sorted(EXCLUDED_SUBCODES),
        "excluded_subcode_items": len(read_jsonl(source)) - len(rows),
        "duplicate_rule": "NFKC/strip/casefold per language; connected components; drop all conflicting label/category/split groups; otherwise keep smallest item_id",
        "conflicting_duplicate_items_removed": conflicts, "redundant_duplicate_items_removed": redundant,
        "n_items": len(retained), "by_source_lang": counts("source_lang"),
        "by_label": {"left": sum(r["label"] == 0 for r in retained), "right": sum(r["label"] == 1 for r in retained)},
        "by_split": {"development": sum(not r["held_out"] for r in retained),
                     "category_holdout": sum(r["held_out"] for r in retained)},
        "by_category": counts("code"), "n_manifestos": len({r["manifesto_id"] for r in retained}),
        "n_parties": len({r["party"] for r in retained}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="build corrected candidate from finalized RILE-v1")
    args = parser.parse_args()
    if args.v2:
        build_v2()
        return
    items = {row["item_id"]: row for row in read_jsonl(DATA / "items.jsonl")}
    translations = {row["item_id"]: row for row in read_jsonl(DATA / "translations.jsonl")}
    for fix in read_jsonl(DATA / "fixes.jsonl"):
        translations[fix["item_id"]][fix["lang"]] = fix["text"]

    decisions = read_jsonl(DATA / "decisions.jsonl")
    dropped = {row["item_id"] for row in decisions if row["action"] == "drop_item"}
    reverts = {(row["item_id"], row["lang"]) for row in decisions if row["action"] == "revert"}
    originals = {row["item_id"]: row for row in read_jsonl(DATA / "translations.jsonl")}

    out = DATA / "final_translations.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for item_id, item in items.items():
            if item_id in dropped:
                continue
            translated = translations[item_id]
            for row_id, lang in reverts:
                if row_id == item_id:
                    translated[lang] = originals[item_id][lang]
            row = {**item, **{lang: translated[lang] for lang in LANGUAGES}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items) - len(dropped)} items; dropped {len(dropped)}; languages: {', '.join(LANGUAGES)}")
    print(out)


if __name__ == "__main__":
    main()
