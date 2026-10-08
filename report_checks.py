# Python checks on the Explanation Agent's report.
#
# count_report_words is an exact check.
# screen_report only FLAGS text for a human to review: matching words
# and numbers cannot confirm that an interpretation is correct.

import re

from feature_dictionary import FEATURE_NAMES


# ---------------------------------------------
# HEADINGS AND WORD COUNT
# ---------------------------------------------

# A heading line may come back decorated, e.g. "## SUMMARY" or
# "**SUMMARY**:". Strip the decoration before comparing.


def normalize_line(line):

    text = line.strip()
    text = re.sub(r"^#+\s*", "", text)
    # "1. SUMMARY" still counts as the heading, so words are not miscounted.
    text = re.sub(r"^\**\d+[.)]\s*", "", text)
    text = text.strip("*_ ").rstrip(":").strip("*_ ")

    return re.sub(r"\s+", " ", text).casefold()


def is_heading(line, headings):
    return normalize_line(line) in {normalize_line(h) for h in headings}


def is_title(line, title):
    return bool(line.strip()) and normalize_line(line) == normalize_line(title)


# Words = whitespace-separated tokens containing a letter or digit.
# The title and the headings are not counted.


def count_words(text):
    return sum(1 for token in text.split() if re.search(r"[A-Za-z0-9]", token))


def split_sections(report, title, headings):

    sections = {"_before_first_heading": []}
    current = "_before_first_heading"

    for line in report.splitlines():

        if is_title(line, title) and current == "_before_first_heading":
            continue

        if is_heading(line, headings):
            match = next(h for h in headings if normalize_line(h) == normalize_line(line))
            current = match
            sections.setdefault(current, [])
            continue

        sections[current].append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def count_report_words(report, title, headings):

    sections = split_sections(report, title, headings)

    by_section = {name: count_words(text) for name, text in sections.items()}

    return {"total": sum(by_section.values()), "by_section": by_section}


# ---------------------------------------------
# SCREENING FLAGS (for human review)
# ---------------------------------------------


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def sentences_matching(text, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    return [sentence for sentence in split_sentences(text) if regex.search(sentence)]


# Decimal numbers and percentages in the report that do not appear in
# the prompt. Percentages are matched against supplied proportions.


def untraceable_numbers(report, prompt):

    supplied = {abs(float(value)) for value in re.findall(r"-?\d+\.\d+|-?\d+", prompt)}

    flagged = []

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(%|percent)|(\d+\.\d+)", report):

        if match.group(3):
            value = float(match.group(3))
            if not any(abs(value - s) < 1e-9 for s in supplied):
                flagged.append(match.group(0))
            continue

        value = float(match.group(1))
        decimals = len(match.group(1).split(".")[1]) if "." in match.group(1) else 0
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9

        if not any(abs(value - 100 * s) <= tolerance for s in supplied if s <= 1):
            flagged.append(match.group(0))

    return sorted(set(flagged))


def contradicts_error_logic(sentence):

    text = sentence.casefold()

    groups = [g for g in ("false positive", "false negative") if g in text]
    directions = [d for d in ("toward decline", "toward stability") if d in text]
    supports = bool(re.search(r"\bsupport", text))
    opposes = bool(re.search(r"\boppos", text))

    # Only unambiguous sentences: one group, one direction, one verb.
    if len(groups) != 1 or len(directions) != 1 or supports == opposes:
        return False

    toward_prediction = (groups[0], directions[0]) in {
        ("false positive", "toward decline"),
        ("false negative", "toward stability"),
    }

    return toward_prediction != supports


def duplicated_rids(package):
    return [
        rid
        for check in package.get("data_checks") or []
        for rid in check.get("duplicated_rids", [])
    ]


# Screening of the report. Every entry is a pointer for manual review,
# not a verdict about correctness. Without headings (or when a heading
# is missing) the section checks below search the whole report.


def screen_report(report, prompt, package, title, headings=()):

    sections = split_sections(report, title, headings)
    wrong_section = sections.get("WHEN THE MODEL GETS IT WRONG") or report

    codes = [
        code
        for code, name in FEATURE_NAMES.items()
        if code != name and re.search(rf"(?<![\w-]){re.escape(code)}(?![\w-])", report)
    ]

    flags = {
        "untraceable_numbers": untraceable_numbers(report, prompt),
        "feature_codes_used": codes,
        "outcome_abbreviations_outside_headings": sorted(
            set(
                re.findall(
                    r"\b(TP|TN|FP|FN)\b",
                    "\n".join(
                        line for line in report.splitlines() if not is_heading(line, headings)
                    ),
                )
            )
        ),
        "parentheses_used": "(" in report or ")" in report,
        # Data field names, raw (average_push) or as a label followed
        # by its value (average push 0.23).
        "field_labels_used": sorted(
            set(re.findall(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b", report))
            | {
                match.group(0)
                for match in re.finditer(
                    r"\b(average push size|average push|push sd|share toward "
                    r"(?:decline|stability)|share neutral|rank difference|"
                    r"passes stability check|mean total push toward "
                    r"(?:decline|stability)|mean net push|n records)"
                    r"\s*[:=]?\s*[-−]?\d",
                    report,
                    re.IGNORECASE,
                )
            }
        ),
        "participant_identifiers": list(
            dict.fromkeys(
                sentences_matching(
                    report,
                    r"\bRIDs?\b|\b(?:participant|patient|subject|record) (?:ID|number)?\s*#?\d{3,}",
                )
                + [
                    sentence
                    for rid in duplicated_rids(package)
                    for sentence in sentences_matching(report, rf"(?<![\d.]){rid}(?![\d.])")
                ]
            )
        ),
        "dominance_language": sentences_matching(
            report, r"\b(overr(ode|ide|ides|idden)|outweigh\w*|dominat\w*|overwhelm\w*)\b"
        ),
        "universal_language": sentences_matching(
            report, r"\b(all|every|always|consistently|invariably|none)\b"
        ),
        "possible_raw_value_claims": sentences_matching(
            report,
            r"\b(high|higher|low|lower|elevated|reduced|poor|poorer|worse|better|"
            r"more|fewer|less)\s+((\w+\s+){0,4}(scores?|volumes?|times?|values?|"
            r"counts?|alleles?|atrophy|results?)\b|("
            + "|".join(re.escape(name) for name in FEATURE_NAMES.values())
            + r"))",
        ),
        "mechanism_or_citation_language": sentences_matching(
            report, r"\b(mechanism|causes?|caused|because of|et al|study shows|studies show)\b"
        ),
    }

    # Comparative words placed directly before a feature name read as a
    # claim about the raw value, e.g. "a higher Feature A".
    names = sorted({name for name in FEATURE_NAMES.values()}, key=len, reverse=True)
    name_pattern = "|".join(re.escape(name) for name in names)
    flags["possible_raw_value_claims"] += [
        sentence
        for sentence in sentences_matching(
            report,
            rf"\b(high|higher|low|lower|elevated|reduced|greater|stronger|weaker)\s+"
            rf"(the\s+)?({name_pattern})",
        )
        if sentence not in flags["possible_raw_value_claims"]
    ]

    # An error group described with the prediction it did NOT get.
    flags["error_group_prediction_contradictions"] = [
        sentence
        for sentence in split_sentences(report)
        if (
            re.search(r"false negative", sentence, re.IGNORECASE)
            and re.search(r"predicted (a )?decline|predicted to decline", sentence, re.IGNORECASE)
        )
        or (
            re.search(r"false positive", sentence, re.IGNORECASE)
            and re.search(r"predicted (to remain |to stay )?stab", sentence, re.IGNORECASE)
        )
    ]

    # For an error group, a push toward the (wrong) predicted class
    # supports the error: false positives were predicted to decline,
    # false negatives to stay stable. Flags a sentence that names one
    # error group and one direction but the opposite verb.
    flags["support_oppose_contradictions"] = [
        sentence
        for sentence in split_sentences(report)
        if contradicts_error_logic(sentence)
    ]

    # Heuristic coverage of the four error groups: paragraphs of the
    # error section that name a cohort and an error type.
    cohort_terms = {
        "CN": r"cognitively normal|\bCN\b",
        "MCI": r"mild cognitive impairment|\bMCI\b",
    }
    error_terms = {
        "false positive": r"false positive|wrongly flagged|flagged as declin|predicted to decline but",
        "false negative": r"false negative|missed|predicted (to remain )?stab\w* but|did not flag",
    }

    covered = set()

    for paragraph in re.split(r"\n\s*\n", wrong_section):
        cohorts = [c for c, p in cohort_terms.items() if re.search(p, paragraph, re.IGNORECASE)]
        errors = [e for e, p in error_terms.items() if re.search(p, paragraph, re.IGNORECASE)]
        covered |= {(c, e) for c in cohorts for e in errors}

    if package.get("has_cohorts"):
        flags["error_groups_not_found_heuristic"] = [
            f"{c} {e}" for c in cohort_terms for e in error_terms if (c, e) not in covered
        ]

    flags["note"] = (
        "Screening only: these flags point a reviewer to sentences worth "
        "checking; an empty flag does not show the interpretation is correct."
    )

    return flags
