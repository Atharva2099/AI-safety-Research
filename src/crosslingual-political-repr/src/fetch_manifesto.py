"""Pull coded Manifesto Project quasi-sentences (text + category code) via the official API.

Needs a free account: https://manifesto-project.wzb.eu/signup -> profile page -> generate API key.
The key is read from MANIFESTO_API_KEY and is never printed or written to disk.

    export MANIFESTO_API_KEY=...
    uv run --with requests python -m src.fetch_manifesto --countries "United Kingdom" "United States" --limit 6

Output: data/manifesto_raw.json (one record per quasi-sentence). The Manifesto Project forbids
redistributing their data, so that path is gitignored: never commit it or upload it to HuggingFace.
"""
import argparse
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://manifesto-project.wzb.eu/tools"


def api(endpoint, api_key, tries=4, **params):
    """GET one endpoint, retrying dropped connections (the server resets long corpus requests fairly often)."""
    for attempt in range(tries):
        try:
            r = requests.get(f"{BASE}/{endpoint}.json", params={"api_key": api_key, **params}, timeout=90)
        except requests.exceptions.RequestException as e:
            if attempt == tries - 1:
                raise
            print(f"  {endpoint}: {type(e).__name__}, retrying ({attempt + 1}/{tries - 1})")
            time.sleep(3 * (attempt + 1))
            continue
        if not r.ok:
            raise SystemExit(f"{endpoint} failed [{r.status_code}]: {r.text[:300]}")
        return r.json()


def core_rows(payload):
    """The core dataset comes back as rows with a header row first; be tolerant about the wrapper."""
    data = payload if isinstance(payload, list) else (payload.get("content") or payload.get("items"))
    header, rows = data[0], data[1:]
    return [dict(zip(header, r)) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--countries", nargs="+", default=["United Kingdom"], help="country names as in the dataset")
    p.add_argument("--limit", type=int, default=6, help="how many party-election documents to pull")
    p.add_argument("--core-version", default="MPDS2026a", help="main dataset version")
    p.add_argument("--corpus-version", default="2026-1", help="corpus/metadata version")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "manifesto_raw.json")
    args = p.parse_args()

    api_key = os.environ.get("MANIFESTO_API_KEY")
    if not api_key:
        raise SystemExit("Set MANIFESTO_API_KEY first (see the module docstring).")

    rows = core_rows(api("api_get_core", api_key, key=args.core_version))
    names = {r.get("countryname") for r in rows}
    wanted = [r for r in rows if r.get("countryname") in args.countries]
    if not wanted:
        raise SystemExit(f"No rows for {args.countries}. Available examples: {sorted(n for n in names if n)[:15]}")
    wanted.sort(key=lambda r: str(r.get("date")))
    print(f"{len(wanted)} party-election rows; taking the {args.limit} most recent")

    docs, skipped = [], []
    for row in wanted[-args.limit:]:
        pid = f"{row['party']}_{str(row['date'])[:6]}"
        try:
            meta = api("api_metadata", api_key, **{"keys[]": pid, "version": args.corpus_version})
            items = meta.get("items") or []
            mid = items[0].get("manifesto_id") if items else None
            if not mid or not items[0].get("annotations"):
                skipped.append(pid)
                continue
            texts = api("api_texts_and_annotations", api_key, **{"keys[]": mid, "version": args.corpus_version})
        except requests.exceptions.RequestException as e:
            print(f"  {pid}: gave up ({type(e).__name__}); keeping what we have")
            break
        for item in texts.get("items") or []:
            for qs in item.get("items") or []:
                docs.append({"manifesto_id": mid, "party": row["party"], "partyname": row.get("partyname"),
                             "date": row["date"], "text": qs.get("text"), "cmp_code": qs.get("cmp_code")})
        print(f"  {pid} {row.get('partyname', '')}: {len(docs)} quasi-sentences so far")

    if skipped:
        print(f"skipped (no annotated text): {', '.join(skipped)}")
    args.out.write_text(json.dumps(docs, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {len(docs)} quasi-sentences to {args.out}")


if __name__ == "__main__":
    main()
