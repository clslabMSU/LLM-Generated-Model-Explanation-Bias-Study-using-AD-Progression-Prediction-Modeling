import json
import os
from datetime import datetime, timezone

import llm_client
from feature_dictionary import describe_features_by_name, feature_name
from prompts.explanation_prompt import (
    PROMPT_VERSION,
    get_cohort_note,
    get_explanation_prompt,
    get_report_title,
)
from report_checks import count_report_words, screen_report


# Keys whose string values (or list of strings) are feature codes.
FEATURE_VALUE_KEYS = {"feature", "columns"}


# Replace feature codes with their full names everywhere in a package
# section, so the LLM never sees "CDRSB".


def to_display(value, parent_key=None):

    if isinstance(value, dict):
        return {
            feature_name(key): to_display(item, key) for key, item in value.items()
        }

    if isinstance(value, list):
        return [to_display(item, parent_key) for item in value]

    if parent_key in FEATURE_VALUE_KEYS and isinstance(value, str):
        return feature_name(value)

    return value


def get_explained_method(explanation_package):

    return (
        explanation_package.get("explained_estimator")
        or explanation_package.get("explained_method")
        or "the best model reported in this study"
    )


def build_explanation_prompt(explanation_package):

    unit = explanation_package["unit"]
    unit_label = explanation_package["unit_label"]

    features = explanation_package.get("features", [])

    unit_lines = "\n".join(f"{column}: {value}" for column, value in unit.items())

    feature_lines = "\n".join(
        f"- {name}: {description}"
        for name, description in describe_features_by_name(features).items()
    )

    def to_json(key):
        # No indentation: the same data in far fewer tokens.
        return json.dumps(
            to_display(explanation_package.get(key)),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return get_explanation_prompt(
        explained_method=get_explained_method(explanation_package),
        unit_lines=unit_lines,
        unit_label=unit_label,
        n_patients=explanation_package.get("n_patients"),
        n_correct=explanation_package.get("n_correct"),
        base_value=explanation_package.get("base_value"),
        feature_dictionary=feature_lines,
        cohort_note=get_cohort_note(explanation_package.get("has_cohorts", False)),
        global_importance_json=to_json("global_importance"),
        confusion_matrix_json=to_json("confusion_matrix"),
        records_json=to_json("records_by_group"),
        group_shap_json=to_json("group_shap"),
        test_metrics_json=to_json("test_metrics"),
        n_unique_rids=explanation_package.get("n_unique_rids"),
    )


def call_llm(ask_llm, prompt):

    started = datetime.now(timezone.utc).isoformat()
    response = (ask_llm(prompt) or "").strip()

    return response, {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_characters": len(prompt),
        **dict(llm_client.LAST_CALL_INFO),
    }


def write_text(folder, name, text):
    with open(os.path.join(folder, name), "w", encoding="utf-8") as file:
        file.write(text)


def save_run(run_dir, record, prompt, package, report):

    os.makedirs(run_dir, exist_ok=True)

    write_text(run_dir, "prompt.md", prompt)

    with open(os.path.join(run_dir, "input_payload.json"), "w", encoding="utf-8") as file:
        json.dump(package, file, indent=2, default=str, ensure_ascii=False)

    write_text(run_dir, "report.md", report)

    with open(os.path.join(run_dir, "run_record.json"), "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, default=str, ensure_ascii=False)


def explain_unit(explanation_package, ask_llm, model_name=None, run_dir=None):

    prompt = build_explanation_prompt(explanation_package)

    title = get_report_title(get_explained_method(explanation_package))

    # The data checks are for the user only; the LLM does not see them.
    print("\nDATA CHECKS:")
    for check in explanation_package.get("data_checks") or []:
        print(f"- [{check['status']}] {check['check']}: {check['detail']}")

    report, call = call_llm(ask_llm, prompt)

    # No headings any more: every word except the title line is counted.
    word_count = count_report_words(report, title, [])["total"]

    # An empty response (e.g. the token limit was spent on thinking) is
    # recorded as a failed generation.
    if report:
        status = "report produced"
    else:
        status = (
            "FAILED: empty response, done_reason "
            f"{call.get('done_reason')}; no report was produced"
        )

    record = {
        "prompt_version": PROMPT_VERSION,
        "model": model_name,
        "generation_settings": (
            llm_client.get_generation_settings(model_name)
            if model_name in llm_client.MODEL_CONFIGS
            else "not recorded: unknown model name"
        ),
        "unit": explanation_package.get("unit_label"),
        "explained_method": explanation_package.get("explained_method"),
        "explained_estimator": explanation_package.get("explained_estimator"),
        "data_checks": explanation_package.get("data_checks"),
        "llm_call": call,
        "status": status,
        "word_count": word_count,
        "word_count_rule": "tokens containing a letter or digit, excluding the title line",
        "screening": screen_report(report, prompt, explanation_package, title),
    }

    print(f"\nReport words: {word_count} ({status})")

    # Screening flags for manual review, shown on the console because
    # the run record may not be saved.
    flagged = {
        name: value
        for name, value in record["screening"].items()
        if name != "note" and value
    }
    if flagged:
        print("\nSCREENING FLAGS (review these sentences; not a verdict):")
        for name, value in flagged.items():
            print(f"- {name}: {value}")

    if run_dir:
        save_run(run_dir, record, prompt, explanation_package, report)
        print("Saved run record to:", run_dir)

    return prompt, report
