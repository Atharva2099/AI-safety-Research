"""Extract available JobPosting fields from collected raw HTML."""

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).parent
RAW = ROOT / "raw"
SOURCES = {row["id"]: row for row in json.loads((ROOT / "sources.json").read_text())}
OUTPUT = ROOT / "parsed_jds.jsonl"

PLAIN_TEXT_HEADINGS = {
    "about the role",
    "about you",
    "how to apply",
    "key responsibilities",
    "nice to have",
    "nice-to-have",
    "nice-to-have technical skills",
    "qualifications",
    "required soft skills",
    "required technical skills",
    "requirements",
    "responsibilities",
    "what you'll bring",
    "what you’ll bring",
    "what you'll do",
    "what you’ll do",
    "what we offer",
    "why join us",
    "✓ you'll thrive here if you...",
    "✗ this role is not for you if you...",
}


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.ignored:
            self.ignored -= 1

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)

    def text(self):
        return " ".join(" ".join(self.parts).split())


class SectionParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ignored = 0
        self.capture = None
        self.parts = []
        self.heading = "preamble"
        self.sections = {self.heading: []}

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.ignored += 1
        elif not self.ignored and tag in {"h1", "h2", "h3", "h4", "p", "li"}:
            self.capture = tag
            self.parts = []

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.ignored:
            self.ignored -= 1
        elif tag == self.capture:
            text = " ".join(" ".join(self.parts).split())
            normalized = text.lower().rstrip(":?")
            if text and tag.startswith("h"):
                self.heading = normalized
                self.sections.setdefault(self.heading, [])
            elif text and len(text) <= 100 and normalized in PLAIN_TEXT_HEADINGS:
                self.heading = normalized
                self.sections.setdefault(self.heading, [])
            elif text and tag == "p" and (
                (len(text) <= 100 and (text.endswith(":") or (text.endswith("?") and text.upper() == text)))
                or normalized.startswith("after you submit your application")
            ):
                self.heading = normalized
                self.sections.setdefault(self.heading, [])
            elif text:
                self.sections.setdefault(self.heading, []).append(text)
            self.capture = None
            self.parts = []

    def handle_data(self, data):
        if not self.ignored and self.capture:
            self.parts.append(data)


def posting_from_json(value):
    if isinstance(value, dict):
        if value.get("@type") == "JobPosting":
            return value
        for child in value.get("@graph", []):
            posting = posting_from_json(child)
            if posting:
                return posting
    if isinstance(value, list):
        for child in value:
            posting = posting_from_json(child)
            if posting:
                return posting
    return None


def extract_posting(raw):
    scripts = re.findall(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for script in scripts:
        try:
            posting = posting_from_json(json.loads(html.unescape(script)))
        except json.JSONDecodeError:
            continue
        if posting:
            return posting
    return None


def clean_description(value):
    parser = TextParser()
    parser.feed(html.unescape(value or ""))
    return parser.text()


def visible_text(raw):
    parser = TextParser()
    parser.feed(raw)
    return parser.text()


def extract_sections(raw):
    parser = SectionParser()
    parser.feed(html.unescape(raw))
    return {heading: values for heading, values in parser.sections.items() if values}


def heading_matches(heading, keywords):
    return any(
        heading == keyword[1:] if keyword.startswith("=") else keyword in heading
        for keyword in keywords
    )


def matching_sections(sections, keywords):
    return [
        text
        for heading, values in sections.items()
        if heading_matches(heading, keywords)
        for text in values
    ]


def select_sections(sections, keywords):
    return {
        heading: values
        for heading, values in sections.items()
        if heading_matches(heading, keywords)
    }


def matching_evidence(sections, pattern):
    return [
        text
        for values in sections.values()
        for text in values
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    selected_ids = set(args.ids or SOURCES)

    with args.output.open("w") as output:
        for path in sorted(RAW.glob("*.html")):
            record_id = path.stem
            if record_id not in selected_ids:
                continue
            raw = path.read_text(errors="replace")
            posting = extract_posting(raw)
            source_html = posting.get("description", "") if posting else raw
            sections = extract_sections(source_html)
            qualification_headings = [
                "requirements", "qualifications", "qualifikation", "who you are",
                "you may be a good fit", "strong candidates", "candidates need not",
                "minimum qualifications", "logistics", "you will need", "you'll thrive",
                "you’ll thrive", "we are looking for team members", "we'd love to hear",
                "we’d love to hear", "what you'll bring", "what you’ll bring", "must have", "muss",
                "=about you",
            ]
            qualification_sections = select_sections(sections, qualification_headings)
            core_qualification_sections = select_sections(
                sections,
                [
                    "requirements", "qualifications", "qualifikation", "who you are",
                    "you may be a good fit", "strong candidates", "minimum qualifications",
                    "you will need", "you'll thrive", "you’ll thrive", "we are looking for team members",
                    "we'd love to hear", "we’d love to hear", "what you'll bring", "what you’ll bring",
                    "must have", "muss", "=about you",
                ],
            )
            evidence = {
                "responsibility_evidence": matching_sections(
                    sections,
                    [
                        "responsibilities", "what you'll do", "what you’ll do", "aufgaben", "your role",
                        "deine rolle", "about the role", "representative projects", "day-to-day, you will",
                        "owning that looks like day to day", "startphase", "im weiteren verlauf",
                    ],
                ),
                "required_qualification_evidence": matching_sections(
                    sections,
                    [
                        "requirements", "qualifications", "qualifikation", "who you are", "you may be a good fit",
                        "minimum qualifications", "you will need", "you'll thrive", "you’ll thrive",
                        "we are looking for team members", "we'd love to hear", "we’d love to hear",
                        "what you'll bring", "what you’ll bring", "must have", "muss", "=about you",
                    ],
                ),
                "preferred_qualification_evidence": matching_sections(
                    sections,
                    [
                        "preferred", "strong candidates may also", "nice to have", "nice-to-have", "bonus",
                        "wünschenswert", "added plus", "wunsch",
                    ],
                ),
                "education_evidence": matching_evidence(
                    qualification_sections, r"\b(degree|bachelor|master|ph\.?d|education|studium|ausbildung|abschluss)\b"
                ),
                "experience_evidence": matching_evidence(
                    qualification_sections, r"\b(experience|years?|erfahrung|berufserfahrung|jahre?)\b"
                ),
                "skill_evidence": matching_evidence(
                    core_qualification_sections,
                    r"\b(skill|knowledge|proficien|familiar|ability|understanding|communication|analytical|kenntnisse|fähigkeit|verständnis|umgang|denkweise)\w*\b",
                ),
                "compensation_evidence": matching_evidence(
                    sections, r"\b(salary|compensation|pay range|annual pay|gehalt|vergütung|€|usd|eur|\$)\b"
                ),
                "location_evidence": matching_evidence(
                    sections,
                    r"\b(location(?:-based|:)|remote|hybrid|on-?site|based in|office locations?|remote locations?|standort|home-?office)\b",
                ),
                "work_authorization_evidence": matching_evidence(
                    sections, r"\b(visa|sponsor|work authori[sz]ation|right to work|e-verify|arbeitserlaubnis)\b"
                ),
                "employment_type_evidence": matching_evidence(
                    sections,
                    r"\b(full-?time|part-?time|contract (?:role|position|employment)|permanent|temporary|internship|vollzeit|teilzeit|festanstellung|freelance)\b",
                ),
            }
            if posting is None:
                source = SOURCES[record_id]
                output.write(json.dumps({
                    "id": record_id,
                    "family": source["family"],
                    "subtype": source["subtype"],
                    "title": source["title"],
                    "organization": source["organization"],
                    "source_url": source["url"],
                    "source_tier": source.get("source_tier", "B"),
                    "description_text": visible_text(raw),
                    "sections": sections,
                    **evidence,
                    "structured_data_available": False,
                    "parse_error": None,
                }) + "\n")
                continue
            description = clean_description(posting.get("description"))
            fields = {
                "id": record_id,
                "family": SOURCES[record_id]["family"],
                "subtype": SOURCES[record_id]["subtype"],
                "title": posting.get("title", SOURCES[record_id]["title"]),
                "organization": posting.get("hiringOrganization"),
                "location": posting.get("jobLocation"),
                "employment_type": posting.get("employmentType"),
                "date_posted": posting.get("datePosted"),
                "valid_through": posting.get("validThrough"),
                "salary": posting.get("baseSalary"),
                "education": posting.get("educationRequirements"),
                "experience": posting.get("experienceRequirements"),
                "qualifications": posting.get("qualifications"),
                "skills": posting.get("skills"),
                "responsibilities": posting.get("responsibilities"),
                "description_text": description,
                "sections": sections,
                **evidence,
                "source_url": SOURCES[record_id]["url"],
                "source_tier": SOURCES[record_id].get("source_tier", "B"),
                "structured_data_available": True,
                "parse_error": None,
            }
            output.write(json.dumps(fields) + "\n")
    print(f"Wrote {sum(1 for _ in args.output.open())} parsed records to {args.output}")


if __name__ == "__main__":
    main()
