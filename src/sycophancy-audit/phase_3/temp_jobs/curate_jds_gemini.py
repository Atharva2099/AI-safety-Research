"""Curate explicitly stated job-description evidence with Gemini JSON output."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "gemini-3.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
CATEGORIES = [
    "responsibility", "qualification", "education", "experience", "skill",
    "compensation", "location", "work_authorization", "employment_type", "other",
]

INSTRUCTION = """You curate evidence from a job description for a later, deterministic
job-fit rubric. Extract only statements that are explicitly present in SOURCE TEXT.

Include responsibilities and candidate-relevant requirements: qualifications, education,
experience, skills, location constraints, work authorization, employment type, or
compensation. Do not include benefits, company marketing, application instructions,
equal-opportunity/disability text, or generic culture statements unless they explicitly
state a candidate requirement. Do not infer a qualification from the title or rewrite a
requirement more strongly than written.

For each claim, copy quote as an exact contiguous substring of SOURCE TEXT, including
punctuation and numbers. Do not provide a summary or restatement.
Set requirement_level to required only for explicit required/minimum/must language;
preferred only for explicit preferred/bonus/nice-to-have language; otherwise unclear.
Return no claim if you cannot provide an exact quote.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "requirement_level": {
                        "type": "string", "enum": ["required", "preferred", "unclear"]
                    },
                    "quote": {"type": "string"},
                },
                "required": ["category", "requirement_level", "quote"],
                "additionalProperties": False,
            },
        },
        "ambiguities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims", "ambiguities"],
    "additionalProperties": False,
}


def generate(text, key):
    body = json.dumps({
        "contents": [{"parts": [{"text": INSTRUCTION + "\nSOURCE TEXT:\n" + text}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
        },
    }).encode()
    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 503} or attempt == 4:
                raise RuntimeError(f"Gemini request failed with HTTP {error.code}") from error
            time.sleep(2 ** attempt)
    response_text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(response_text), payload, response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set")

    ids = set(args.ids or [])
    rows = [json.loads(line) for line in args.input.open()]
    if ids:
        rows = [row for row in rows if row["id"] in ids]
    if ids and len(rows) != len(ids):
        raise SystemExit("one or more requested IDs were not found")
    completed = set()
    if args.output.exists():
        completed = {json.loads(line)["id"] for line in args.output.open()}
    with args.output.open("a") as output:
        for row in rows:
            if row["id"] in completed:
                continue
            curated, raw, response_text = generate(row["description_text"], key)
            output.write(json.dumps({
                "id": row["id"],
                "model": MODEL,
                "claims": curated["claims"],
                "ambiguities": curated["ambiguities"],
                "finish_reason": raw["candidates"][0].get("finishReason"),
                "raw_response_text": response_text,
            }) + "\n")
            output.flush()
            print(row["id"])


if __name__ == "__main__":
    main()
