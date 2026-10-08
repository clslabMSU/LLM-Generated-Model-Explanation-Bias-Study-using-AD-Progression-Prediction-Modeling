import json
import re

from llm_client import GPT_MODEL, QWEN_MODEL
from prompts.best_optimizer_prompt import (
    get_decision_model_note,
    get_gpt_decision_prompt,
    get_qwen_decision_prompt,
)

# Each LLM gets its own prompt text, as in the team's code
# (decision_Agent_Qwen.py for Qwen, decision_Agent.py for GPT-OSS).
DECISION_PROMPTS = {
    QWEN_MODEL: get_qwen_decision_prompt,
    GPT_MODEL: get_gpt_decision_prompt,
}


def build_decision_prompt(analysis_package, model_name=QWEN_MODEL):

    # --------------------------------------------------
    # GET THE DATA PREPARED BY THE PYTHON AGENT
    # --------------------------------------------------

    unit = analysis_package["unit"]
    unit_label = analysis_package["unit_label"]
    methods = analysis_package["methods"]

    primary_results = analysis_package["primary_comparison"]

    supporting_results = analysis_package["supporting_results_by_method"]

    explainability = analysis_package.get("explainability")

    # The unit columns depend on the file:
    # Dataset + Timepoint for master_results,
    # Target for ADNI_model_data.
    unit_lines = "\n".join(f"{column}: {value}" for column, value in unit.items())

    method_lines = "\n".join(f"- {method}" for method in methods)

    method_count = len(methods)

    # The unit fields are echoed back in the JSON answer.
    unit_json_fields = "".join(
        f'  "{column.lower()}": "{value}",\n' for column, value in unit.items()
    )

    # How many folds each method has in this unit.
    fold_counts = {method: len(rows) + 1 for method, rows in supporting_results.items()}

    max_folds = max(fold_counts.values()) if fold_counts else 0

    # Convert the Python dictionaries/lists into
    # JSON-formatted text for the LLM.
    primary_results_json = json.dumps(primary_results, indent=2)

    supporting_results_json = json.dumps(supporting_results, indent=2)

    # --------------------------------------------------
    # FEATURE IMPORTANCE SECTION
    # --------------------------------------------------
    # Only built when the file carries PFI / SHAP sheets.
    # They describe one method only, so the section states
    # which one and the answer is conditional on it.

    explained_method = None
    explainability_section = ""
    feature_explanation_field = ""
    feature_explanation_rule = ""

    if explainability:

        explained_method = explainability.get("explained_method")

        explainability_json = json.dumps(explainability, indent=2)

        explained_method_text = (
            explained_method
            if explained_method
            else "the best model reported in this study"
        )

        # As in the team's code, explainability_section and
        # feature_explanation_rule stay empty: feature importance is
        # left to the explanation agent.

    # --------------------------------------------------
    # DECISION AGENT PROMPT (text in prompts/best_optimizer_prompt.py)
    # --------------------------------------------------

    get_prompt = DECISION_PROMPTS.get(model_name, get_qwen_decision_prompt)

    prompt = get_prompt(
        unit_lines=unit_lines,
        unit_label=unit_label,
        method_count=method_count,
        method_lines=method_lines,
        max_folds=max_folds,
        primary_results_json=primary_results_json,
        supporting_results_json=supporting_results_json,
        explainability_section=explainability_section,
        feature_explanation_rule=feature_explanation_rule,
    )

    return prompt + get_decision_model_note(model_name)


# Caption lines of the team's prompt template that the LLM sometimes
# copies into the report. Removed here so the prompt text itself can
# stay identical to the team's.
TEMPLATE_CAPTIONS = [
    "*Rank-1 values copied exactly from <PRIMARY_RESULTS>; no new averages.*",
]

# "**Selected method:** TPOT" and the like. Consecutive field lines
# need two trailing spaces, or Markdown joins them into one line.
FIELD_LINE = re.compile(r"^\*\*[^*]+:\*\*")

# ask_llm is the function that calls the LLM,
# e.g. ask_qwen or ask_llama from llm_client.
# model_name picks the prompt text and is named in Provenance.


def analyze_unit(analysis_package, ask_llm, model_name=QWEN_MODEL):

    prompt = build_decision_prompt(analysis_package, model_name)

    response = ask_llm(prompt)

    return response
