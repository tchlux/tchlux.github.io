from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CARDS_MD = ROOT / "cards.md"
EXAM_MD = ROOT / "pa_psych_law_mock_exam.md"
SOURCE_MAP_MD = ROOT / "pa_psych_law_source_map.md"
INDEX_HTML = ROOT / "index.html"
START_MARKER = "<!-- CARDS_MARKDOWN:START -->"
END_MARKER = "<!-- CARDS_MARKDOWN:END -->"


# Read a UTF-8 text file from the workspace.
#
# Arguments:
#   path (Path): file path to read
# 
# Returns:
#   (str): full file contents
#
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Write a UTF-8 text file only when contents changed.
#
# Arguments:
#   path (Path): file path to write
#   text (str): new file contents
# 
# Returns:
#   (bool): whether the file changed
#
def write_text(path: Path, text: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


# Parse canonical source-map metadata by chunk id.
#
# Arguments:
#   raw (str): source-map markdown contents
# 
# Returns:
#   (dict[str, dict[str, str]]): chunk metadata keyed by source chunk id
#
def parse_source_map(raw: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r"^- `(?P<chunk>[AB]\d{2})` \| `(?P<source_file>[^`]+)` \| `(?P<citation>[^`]+)` \| `(?P<category>[^`]+)` \| (?P<summary>.+)$",
        re.M,
    )
    for match in pattern.finditer(raw):
        data = match.groupdict()
        chunk = data["chunk"]
        # Store the normalized app fields for this chunk.
        entries[chunk] = {
            "source_file": data["source_file"],
            "citation": data["citation"],
            "category": data["category"],
            "source_quote": data["summary"].strip(),
        }
    if len(entries) != 120:
        raise ValueError(f"Expected 120 source-map entries, found {len(entries)}.")
    return entries


# Parse the question stem and four choices for every question.
#
# Arguments:
#   raw (str): mock-exam markdown contents
# 
# Returns:
#   (dict[int, dict[str, object]]): question text and choices keyed by id
#
def parse_questions(raw: str) -> dict[int, dict[str, object]]:
    try:
        body = raw.split("## Questions", 1)[1].split("## Answer Key", 1)[0]
    except IndexError as exc:
        raise ValueError("Could not find Questions and Answer Key sections.") from exc
    questions: dict[int, dict[str, object]] = {}
    pattern = re.compile(
        r"(?ms)^\s*(?P<id>\d+)\. (?P<question>.+?)\n"
        r"A\. (?P<a>.+?)\n"
        r"B\. (?P<b>.+?)\n"
        r"C\. (?P<c>.+?)\n"
        r"D\. (?P<d>.+?)\n(?=\n\d+\. |\n## |\Z)"
    )
    for match in pattern.finditer(body):
        qid = int(match.group("id"))
        # Preserve the exact question text and choice wording from the exam file.
        questions[qid] = {
            "question": match.group("question").strip(),
            "choices": [
                match.group("a").strip(),
                match.group("b").strip(),
                match.group("c").strip(),
                match.group("d").strip(),
            ],
        }
    if len(questions) != 120:
        raise ValueError(f"Expected 120 questions, found {len(questions)}.")
    return questions


# Parse the answer key into answers, rationales, and chunk ids.
#
# Arguments:
#   raw (str): mock-exam markdown contents
# 
# Returns:
#   (dict[int, dict[str, str]]): answer-key data keyed by question id
#
def parse_answer_key(raw: str) -> dict[int, dict[str, str]]:
    try:
        body = raw.split("## Answer Key", 1)[1]
    except IndexError as exc:
        raise ValueError("Could not find Answer Key section.") from exc
    answers: dict[int, dict[str, str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("- `Q"):
            continue
        match = re.match(r"- `Q(?P<id>\d+): (?P<answer>[ABCD])` — (?P<rest>.+)", line)
        if not match:
            raise ValueError(f"Malformed answer-key line: {line}")
        qid = int(match.group("id"))
        rationale, source = match.group("rest").rsplit(" Source: ", 1)
        source_match = re.fullmatch(r"`(?P<chunk>[AB]\d{2})`, `(?P<citation>.+)`\.", source)
        if not source_match:
            raise ValueError(f"Malformed source reference in answer key: {line}")
        # Keep the rationale from the exam file and join later with source-map metadata.
        answers[qid] = {
            "answer": match.group("answer"),
            "rationale": rationale.strip(),
            "source_chunk_id": source_match.group("chunk"),
            "answer_citation": source_match.group("citation"),
        }
    if len(answers) != 120:
        raise ValueError(f"Expected 120 answer-key entries, found {len(answers)}.")
    return answers


# Build the normalized card list from the two legacy markdown sources.
#
# Arguments:
#   questions (dict[int, dict[str, object]]): parsed question stems and choices
#   answers (dict[int, dict[str, str]]): parsed answer-key data
#   source_map (dict[str, dict[str, str]]): parsed source-map metadata
# 
# Returns:
#   (list[dict[str, object]]): normalized app cards in question order
#
def build_cards(
    questions: dict[int, dict[str, object]],
    answers: dict[int, dict[str, str]],
    source_map: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    seen_chunks: set[str] = set()
    for qid in range(1, 121):
        question = questions.get(qid)
        answer = answers.get(qid)
        if not question or not answer:
            raise ValueError(f"Missing data for question {qid}.")
        chunk = answer["source_chunk_id"]
        meta = source_map.get(chunk)
        if not meta:
            raise ValueError(f"Question {qid} references unknown chunk {chunk}.")
        if chunk in seen_chunks:
            raise ValueError(f"Duplicate chunk mapping detected for {chunk}.")
        seen_chunks.add(chunk)
        # Use the source-map summary as the initial source excerpt until improved quotes are added.
        cards.append(
            {
                "id": qid,
                "source_chunk_id": chunk,
                "source_file": meta["source_file"],
                "citation": meta["citation"],
                "category": meta["category"],
                "source_quote": meta["source_quote"],
                "question": question["question"],
                "choices": question["choices"],
                "correct_choice": ord(answer["answer"]) - 65,
                "rationale": answer["rationale"],
            }
        )
    return cards


# Render the canonical cards markdown file.
#
# Arguments:
#   cards (list[dict[str, object]]): normalized app cards
# 
# Returns:
#   (str): full canonical markdown contents
#
def format_cards_md(cards: list[dict[str, object]]) -> str:
    blocks = [
        "# PA Psychology Law Study Cards",
        "",
        "Canonical app content. Edit this file, then run `python3 sync_cards.py`.",
        "",
    ]
    for card in cards:
        answer = chr(card["correct_choice"] + 65)
        blocks.extend(
            [
                f"## Card {card['id']}",
                "",
                f"- `id`: `{card['id']}`",
                f"- `source_chunk_id`: `{card['source_chunk_id']}`",
                f"- `source_file`: `{card['source_file']}`",
                f"- `citation`: `{card['citation']}`",
                f"- `category`: `{card['category']}`",
                f"- `answer`: `{answer}`",
                "",
                "### Question",
                str(card["question"]).strip(),
                "",
                "### Choices",
                f"- A. {card['choices'][0]}",
                f"- B. {card['choices'][1]}",
                f"- C. {card['choices'][2]}",
                f"- D. {card['choices'][3]}",
                "",
                "### Rationale",
                str(card["rationale"]).strip(),
                "",
                "### Source Quote",
                str(card["source_quote"]).strip(),
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


# Extract a labeled section body from a card block.
#
# Arguments:
#   block (str): one card block
#   heading (str): section heading to extract
# 
# Returns:
#   (str): extracted section body
#
def section_body(block: str, heading: str) -> str:
    match = re.search(rf"(?ms)^### {re.escape(heading)}\n(.*?)(?=^### |\Z)", block)
    if not match:
        raise ValueError(f"Missing section '{heading}'.")
    return match.group(1).strip()


# Parse and validate the canonical cards markdown file.
#
# Arguments:
#   raw (str): cards markdown contents
# 
# Returns:
#   (list[dict[str, object]]): normalized app cards
#
def parse_cards_md(raw: str) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    ids: set[int] = set()
    chunks: set[str] = set()
    pattern = re.compile(r"(?ms)^## Card (?P<id>\d+)\n(?P<body>.*?)(?=^## Card \d+\n|\Z)")
    for match in pattern.finditer(raw):
        body = match.group("body").strip()
        qid = int(match.group("id"))

        def field(name: str) -> str:
            found = re.search(rf"(?m)^- `{re.escape(name)}`: `(.+?)`$", body)
            if not found:
                raise ValueError(f"Card {qid} is missing field '{name}'.")
            return found.group(1)

        choices = []
        for choice in section_body(body, "Choices").splitlines():
            choice = choice.strip()
            found = re.fullmatch(r"- ([ABCD])\. (.+)", choice)
            if not found:
                raise ValueError(f"Card {qid} has malformed choice line: {choice}")
            choices.append(found.group(2).strip())
        if len(choices) != 4:
            raise ValueError(f"Card {qid} must have exactly 4 choices.")

        chunk = field("source_chunk_id")
        if qid in ids:
            raise ValueError(f"Duplicate card id {qid}.")
        if chunk in chunks:
            raise ValueError(f"Duplicate source chunk {chunk}.")
        ids.add(qid)
        chunks.add(chunk)

        answer = field("answer")
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Card {qid} has invalid answer '{answer}'.")

        # Convert the strict markdown format into the app's runtime shape.
        cards.append(
            {
                "id": int(field("id")),
                "source_chunk_id": chunk,
                "source_file": field("source_file"),
                "citation": field("citation"),
                "category": field("category"),
                "source_quote": section_body(body, "Source Quote"),
                "question": section_body(body, "Question"),
                "choices": choices,
                "correct_choice": ord(answer) - 65,
                "rationale": section_body(body, "Rationale"),
            }
        )
    if not cards:
        raise ValueError("No cards found in cards.md.")
    return cards


# Inject the canonical markdown into the HTML marker block.
#
# Arguments:
#   html (str): current HTML file
#   cards_md (str): canonical cards markdown
# 
# Returns:
#   (str): updated HTML contents
#
def inject_cards_markdown(html: str, cards_md: str) -> str:
    block = (
        f"{START_MARKER}\n"
        f'  <script id="cards-markdown" type="text/plain">\n'
        f"{cards_md.rstrip()}\n"
        f"  </script>\n"
        f"  {END_MARKER}"
    )
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        re.S,
    )
    if not pattern.search(html):
        raise ValueError("index.html is missing card injection markers.")
    return pattern.sub(block, html, count=1)


# Build or refresh cards.md from the legacy markdown source files.
#
# Arguments:
#   force (bool): whether to regenerate even if cards.md already exists
# 
# Returns:
#   (bool): whether cards.md changed
#
def bootstrap_cards(force: bool) -> bool:
    if CARDS_MD.exists() and not force:
        return False
    cards = build_cards(
        parse_questions(read_text(EXAM_MD)),
        parse_answer_key(read_text(EXAM_MD)),
        parse_source_map(read_text(SOURCE_MAP_MD)),
    )
    return write_text(CARDS_MD, format_cards_md(cards))


# Parse CLI args for sync behavior.
#
# Arguments:
#   None
# 
# Returns:
#   (argparse.Namespace): parsed CLI arguments
#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical cards.md and inject it into index.html.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Regenerate cards.md from pa_psych_law_mock_exam.md and pa_psych_law_source_map.md before syncing.",
    )
    return parser.parse_args()


# Sync cards.md into index.html after validation.
#
# Arguments:
#   None
# 
# Returns:
#   (int): process exit code
#
def main() -> int:
    args = parse_args()
    cards_changed = bootstrap_cards(force=args.bootstrap)
    cards_md = read_text(CARDS_MD)
    cards = parse_cards_md(cards_md)
    html_changed = write_text(INDEX_HTML, inject_cards_markdown(read_text(INDEX_HTML), cards_md))
    # Print a short status summary for the caller.
    print(
        f"cards={len(cards)} cards_md={'updated' if cards_changed else 'ok'} "
        f"index_html={'updated' if html_changed else 'ok'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
