"""Download public job-description pages listed in sources.json."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
SOURCES = ROOT / "sources.json"
RAW = ROOT / "raw"
METADATA = ROOT / "raw_metadata.jsonl"


def main():
    sources = json.loads(SOURCES.read_text())
    RAW.mkdir(exist_ok=True)
    with METADATA.open("w") as metadata_file:
        for source in sources:
            record_id = source["id"]
            request = Request(
                source["url"],
                headers={"User-Agent": "AI-safety-research-learning-project/1.0"},
            )
            accessed_at = datetime.now(timezone.utc).isoformat()
            try:
                with urlopen(request, timeout=30) as response:
                    content = response.read()
                    content_type = response.headers.get("Content-Type", "")
                    status = response.status
                (RAW / f"{record_id}.html").write_bytes(content)
                result = {
                    **source,
                    "accessed_at": accessed_at,
                    "status": status,
                    "content_type": content_type,
                    "bytes": len(content),
                    "error": None,
                }
            except Exception as error:
                result = {**source, "accessed_at": accessed_at, "error": repr(error)}
            metadata_file.write(json.dumps(result) + "\n")
            metadata_file.flush()
            time.sleep(1)
            print(result)


if __name__ == "__main__":
    main()
