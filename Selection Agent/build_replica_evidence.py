"""Render six existing model-selection decisions as compact evidence tables.

Example (PowerShell, from the project directory):
    python .\build_replica_evidence.py --root .

Reads decision_results_qwen/run_01..05 and decision_results_gpt/run_01..05.
Accepts Converted_decision.md (any case) or converted_decision.txt.
No models are run and no decisions or numbers are recalculated.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

AGENTS = (("Qwen", "decision_results_qwen"), ("GPT-OSS", "decision_results_gpt"))
RUNS = range(1, 6)
HEADING = re.compile(r"^##[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
TABLE_RULE = re.compile(r"^:?-{3,}:?$")
FIELD = re.compile(r"^\s*\*\*([^:*]+?):\*\*[ \t\u00a0]*(.*?)[ \t]*$", re.MULTILINE)


def choose_source(folder: Path, stem: str) -> Path:
    if not folder.is_dir():
        raise ValueError(f"Missing run folder: {folder}")
    matches = [p for p in folder.iterdir() if p.is_file()
               and p.stem.casefold() == stem.casefold()
               and p.suffix.casefold() in {".md", ".txt"}]
    if not matches:
        raise ValueError(f"No {stem}.md or {stem}.txt in {folder}")
    # The requested Markdown decision takes priority if both formats exist.
    markdown = [p for p in matches if p.suffix.casefold() == ".md"]
    selected = markdown or matches
    if len(selected) != 1:
        raise ValueError(f"Ambiguous decision files in {folder}: {', '.join(map(str, selected))}")
    return selected[0]


def sections_from(text: str, path: Path) -> dict[str, str]:
    headings = list(HEADING.finditer(text))
    sections = {}
    for index, match in enumerate(headings):
        name = match.group(1).strip().casefold()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        if name in sections:
            raise ValueError(f"Repeated ## {name} in {path}")
        # Keep the contents literally as the agent wrote them; only remove
        # blank padding around the heading boundary.
        sections[name] = text[match.end():end].strip("\r\n")
    for required in ("answer", "evidence", "interpretation", "limitations"):
        if not sections.get(required, "").strip():
            raise ValueError(f"Missing or empty ## {required.title()} in {path}")
    return sections


def escape_inline(value: str) -> str:
    """Show Markdown bold as bold without changing the displayed words."""
    escaped = html.escape(value, quote=True)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def pipe_cells(line: str) -> list[str]:
    # Markdown evidence tables in the agent output use plain pipe separators.
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(line: str) -> bool:
    cells = pipe_cells(line)
    return len(cells) >= 2 and all(TABLE_RULE.fullmatch(c) for c in cells)


def render_text(source: str) -> str:
    """Display original words/numbers in paragraphs, lists, and evidence tables."""
    lines = source.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if i + 1 < len(lines) and lines[i].lstrip().startswith("|") and is_separator(lines[i + 1]):
            headers = pipe_cells(lines[i])
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                row = pipe_cells(lines[i])
                if len(row) != len(headers):
                    raise ValueError("Evidence table has a row with a different column count")
                rows.append(row)
                i += 1
            head = "".join(f"<th>{escape_inline(c)}</th>" for c in headers)
            body = "".join("<tr>" + "".join(f"<td>{escape_inline(c)}</td>" for c in row) + "</tr>" for row in rows)
            blocks.append(f'<div class="table-scroll"><table class="metrics"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
            continue
        part = []
        while i < len(lines) and lines[i].strip() and not (
            i + 1 < len(lines) and lines[i].lstrip().startswith("|") and is_separator(lines[i + 1])
        ):
            part.append(escape_inline(lines[i]))
            i += 1
        blocks.append('<p class="source-text">' + "<br>".join(part) + "</p>")
    return "\n".join(blocks)


def display_method_names(source: str, classifier: str | None = None) -> str:
    """Apply the requested display labels without editing the run files."""
    source = source.replace("Manual Random Forest", "Random Forest")
    if classifier:
        source = re.sub(r"\bTPOT\b(?!\s*\()", f"TPOT ({classifier})", source)
    return source


def field_value(source: str, label: str, path: Path) -> tuple[str, str]:
    """Return both the literal field line and its literal value."""
    matches = [(m.group(0).strip(), m.group(2).strip()) for m in FIELD.finditer(source)
               if m.group(1).strip().casefold() == label.casefold()]
    # Some outputs repeat the same coding fields in ## Evaluation. Repetition
    # is harmless, but conflicting values would make extraction unreliable.
    if not matches or not matches[0][1] or any(value != matches[0][1] for _, value in matches):
        raise ValueError(f"Missing or conflicting {label!r} field in {path}")
    return matches[0]


def decision_text(sections: dict[str, str], path: Path, classifier: str | None) -> str:
    answer = sections["answer"]
    selected_line, _ = field_value(answer, "Selected method", path)
    # Keep the agent's answer paragraph(s), excluding the compact metadata
    # lines that the user does not want to display in Decision.
    narrative = "\n".join(line for line in answer.splitlines()
                          if not FIELD.fullmatch(line) and line.strip()).strip()
    # Some valid agent runs provide only selection fields under ## Answer.
    # Show the exact selected-method line in that case; never invent a sentence.
    result = selected_line + ("\n\n" + narrative if narrative else "")
    return display_method_names(result, classifier)


def evidence_table(source: str, path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = source.splitlines()
    tables = []
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("|") and is_separator(lines[i + 1]):
            headers = pipe_cells(lines[i])
            data = []
            for line in lines[i + 2:]:
                if not line.lstrip().startswith("|"):
                    break
                cells = pipe_cells(line)
                if len(cells) != len(headers):
                    raise ValueError(f"Malformed evidence row in {path}")
                data.append(dict(zip(headers, cells)))
            if "Method" in headers:
                tables.append((headers, data))
    if len(tables) != 1:
        raise ValueError(f"Expected one evidence table with a Method column in {path}")
    return tables[0]


def selected_classifier(sections: dict[str, str], path: Path) -> str | None:
    selected = field_value(sections["answer"], "Selected method", path)[1]
    if selected != "TPOT":
        return None
    headers, rows = evidence_table(sections["evidence"], path)
    candidates = []
    if "Estimator" in headers:
        candidates.extend(row["Estimator"] for row in rows if row["Method"] == selected)
    for section in sections.values():
        candidates.extend(value for match in FIELD.finditer(section)
                          if match.group(1).strip().casefold() == "selected estimator"
                          for value in [match.group(2).strip()])
    if not candidates or any(value != candidates[0] for value in candidates) or not candidates[0]:
        raise ValueError(f"TPOT classifier missing or inconsistent in {path}")
    return candidates[0]


def drivers_table(sections: dict[str, str], path: Path, classifier: str | None) -> str:
    selected = field_value(sections["answer"], "Selected method", path)[1]
    runner_up = field_value(sections["answer"], "Runner-up method", path)[1]
    # The prompt has used both 'Agreement coding' and 'Evaluation' for these
    # fields. Search all sections, but require one unambiguous value for each.
    source = "\n".join(sections.values())
    metrics = [field_value(source, f"{ordinal} primary criterion metric", path)[1]
               for ordinal in ("First", "Second")]
    if metrics[0] == metrics[1]:
        raise ValueError(f"Duplicate primary criterion metric in {path}")
    headers, rows = evidence_table(sections["evidence"], path)
    missing = [metric for metric in metrics if metric not in headers]
    if missing:
        raise ValueError(f"Decision metric(s) absent from Evidence table in {path}: {', '.join(missing)}")
    chosen = []
    for method in (selected, runner_up):
        matches = [row for row in rows if row["Method"] == method]
        if len(matches) != 1 or any(not matches[0][metric] for metric in metrics):
            raise ValueError(f"Missing or ambiguous evidence values for {method!r} in {path}")
        chosen.append(matches[0])
    columns = ["Method", *metrics]
    head = "".join(f"<th>{escape_inline(name)}</th>" for name in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape_inline(display_method_names(row[name], classifier) if name == 'Method' else row[name])}</td>" for name in columns)
                   + "</tr>" for row in chosen)
    return f'<div class="table-scroll"><table class="metrics"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def card(agent: str, run: int, path: Path, sections: dict[str, str], root: Path) -> str:
    classifier = selected_classifier(sections, path)
    fields = (("Decision", render_text(decision_text(sections, path, classifier))),
              ("Main Selection Drivers", drivers_table(sections, path, classifier)),
              ("Evidence", render_text(display_method_names(sections["interpretation"]))),
              ("Limitations", render_text(display_method_names(sections["limitations"]))))
    rows = "\n".join(f'<tr><td><strong class="row-label">{title}:</strong>{content}</td></tr>'
                     for title, content in fields)
    return (f'<article class="card"><h2>{html.escape(agent)} · Replica {run:02d}</h2>'
            f'<table class="card-table"><tbody>{rows}</tbody></table></article>')


def build(root: Path, stem: str) -> str:
    cards = []
    for agent, folder in AGENTS:
        for run in RUNS:
            path = choose_source(root / folder / f"run_{run:02d}", stem)
            raw = path.read_text(encoding="utf-8-sig")
            sections = sections_from(raw, path)
            cards.append(card(agent, run, path, sections, root))
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Model-selection replica evidence</title>
<style>
:root { font-family: Arial, sans-serif; color: #1c2733; background: #f5f7fa; }
body { margin: 0 auto; max-width: 1500px; padding: 24px; }
h1 { font-size: 1.6rem; margin: 0 0 8px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.card { min-width: 0; background: white; border: 2px solid #a96409; break-inside: avoid; }
h2 { margin: 0; padding: 10px 12px; color: #fff; background: #a96409; font-size: 1.12rem; }
.card-table { border-collapse: collapse; width: 100%; table-layout: fixed; }
.card-table > tbody > tr > td { border-top: 2px solid #a96409; padding: 9px; }
.row-label { display: block; margin-bottom: 5px; }
.source-text { margin: 0 0 7px; overflow-wrap: anywhere; }
.source-text:last-child { margin-bottom: 0; }
.table-scroll { overflow-x: auto; max-width: 100%; }
.metrics { border-collapse: collapse; font-size: .78rem; white-space: nowrap; }
.metrics th, .metrics td { border: 1px solid #abb7c5; padding: 4px 6px; text-align: left; }
.metrics th { background: #e8eff7; }
@media (max-width: 950px) { .grid { grid-template-columns: 1fr; } }
@media print { :root { background: white; } body { max-width: none; padding: 0; }
  .grid { grid-template-columns: 1fr; gap: 8mm; } .card { break-inside: avoid; }
  .table-scroll { overflow: visible; } .metrics { white-space: normal; font-size: 7pt; }
  @page { size: A3 landscape; margin: 10mm; } }
</style></head><body>
<h1>Model-selection evidence by agent and replica</h1>
<main class="grid">''' + "\n".join(cards) + "</main></body></html>\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Project directory containing both decision_results folders")
    parser.add_argument("--output", type=Path, default=Path("replica_evidence.html"), help="HTML output file")
    parser.add_argument("--stem", default="Converted_decision", help="Decision filename without .md/.txt")
    args = parser.parse_args()
    root = args.root.resolve()
    result = build(root, args.stem)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    print(f"Wrote six evidence cards to {output}")


if __name__ == "__main__":
    main()
