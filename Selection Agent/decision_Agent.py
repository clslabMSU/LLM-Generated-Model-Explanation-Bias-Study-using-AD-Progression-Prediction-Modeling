import json
from pathlib import Path


PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "decision_prompt.txt"


def build_decision_prompt(analysis_package):
    unit = analysis_package["unit"]
    unit_label = analysis_package["unit_label"]
    methods = analysis_package["methods"]
    primary_results = analysis_package["primary_comparison"]
    supporting_results = analysis_package["supporting_results_by_method"]

    unit_lines = "\n".join(
        f"{column}: {value}"
        for column, value in unit.items()
    )
    method_lines = "\n".join(f"- {method}" for method in methods)

    fold_counts = {
        method: len(rows) + 1
        for method, rows in supporting_results.items()
    }
    max_folds = max(fold_counts.values()) if fold_counts else 0

    prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return prompt_template.format(
        unit_lines=unit_lines,
        method_count=len(methods),
        method_lines=method_lines,
        primary_results_json=json.dumps(primary_results, indent=2),
        supporting_results_json=json.dumps(supporting_results, indent=2),
        max_folds=max_folds,
        unit_label=unit_label,
    )


def analyze_unit(analysis_package, ask_model):
    prompt = build_decision_prompt(analysis_package)
    return ask_model(prompt)
