import os
import sys

from data_processing import (
    load_results,
    load_explainability,
    load_predictions,
    get_units,
    get_unit_results,
    get_unit_columns,
    format_unit_label,
    format_unit_filename,
    prepare_analysis_package,
    prepare_explanation_package,
)

from decision_agent import analyze_unit
from explanation_agent import explain_unit
import llm_client
from llm_client import ASK_FUNCTIONS, QWEN_MODEL, GPT_MODEL, get_explanation_ask

# Change this one line to switch LLM (QWEN_MODEL or GPT_MODEL).
# Both agents, the output folder and the output file names follow it.
# In the test runs on ADNI_model_data.xlsx, qwen3 read every SHAP sign
# and comparison correctly in the explanation; gpt-oss either reversed
# conclusions (think "low") or ran out of tokens while thinking
# (think "medium").
MODEL_NAME = GPT_MODEL

ask_llm = ASK_FUNCTIONS[MODEL_NAME]

# The Decision Agent is run and saved by another team member, so it is
# off here: no LLM call and no decision file. Set to True to run it.
RUN_DECISION = False

# Set to False to skip the Explanation Agent.
RUN_EXPLANATION = True

OUTPUT_FOLDERS = {
    "qwen3:30b": "results_qwen",
    "gpt-oss:20b": "results_gpt",
}

# Unknown models get their own folder instead of overwriting another model's results
OUTPUT_FOLDER = OUTPUT_FOLDERS.get(
    MODEL_NAME, "results_" + MODEL_NAME.replace(":", "_")
)

FILE_PATH = "data/ADNI_model_data.xlsx"

# How many times the whole analysis is repeated in one run.
# The file names carry the replicate number, e.g. ..._explanation_rep1.md.
N_REPLICATES = 1


# LLM name safe for file names: "qwen3:30b" -> "qwen3_30b"
# (Windows does not allow ":" in file names).


def get_llm_label(model):
    return format_unit_filename({"llm": model.replace(":", "_")})


# LLM temperature for file names, read from llm_client.TEMPERATURE,
# e.g. 0 -> "temp0", 0.7 -> "temp0.7". Change the temperature there;
# runs at different temperatures then get different file names.


def get_temperature_label():
    return f"temp{llm_client.TEMPERATURE:g}"


# Decision Agent for one unit. Only used when RUN_DECISION is True.


def run_decision(unit_results, unit, explainability, replicate_tag):

    analysis_package = prepare_analysis_package(unit_results, unit, explainability)

    print("\n======================================")
    print("RUNNING DECISION AGENT")
    print(format_unit_label(unit))
    print("======================================")

    decision = analyze_unit(analysis_package, ask_llm, MODEL_NAME)

    print("\nDECISION AGENT RESPONSE:")
    print(decision)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    file_name = f"{format_unit_filename(unit)}_decision_{replicate_tag}.md"
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(decision)

    print("\nSaved decision to:")
    print(output_path)


def main():

    # LLM output contains characters such as non-breaking hyphens that
    # the Windows console encoding cannot print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df = load_results(FILE_PATH)
    df.columns = df.columns.str.strip()

    print("Data got loaded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # PFI / SHAP sheets, if this file has them.
    explainability = load_explainability(FILE_PATH)

    if explainability:
        print("\nExplainability sheets loaded:")
        print(list(explainability.keys()))
    else:
        print("\nNo explainability sheets in this file.")

    # <Method>_Predictions sheets (CN / MCI labels), if any.
    predictions = load_predictions(FILE_PATH)

    if predictions:
        print("\nPrediction sheets loaded:")
        print(list(predictions.keys()))

    unit_columns = get_unit_columns(df)
    units = get_units(df)

    print("\nUnit columns:")
    print(unit_columns)

    print("\nUnits found:")
    for unit in units:
        print(" -", format_unit_label(unit))

    if RUN_DECISION:
        print(f"\nDecision model: {MODEL_NAME} -> {OUTPUT_FOLDER}")
    else:
        print("\nDecision Agent: off (RUN_DECISION = False)")

    if RUN_EXPLANATION:
        print(f"Explanation model: {MODEL_NAME} -> {OUTPUT_FOLDER}")

    print("\nMethods:")
    print(df["Method"].dropna().unique())

    for replicate in range(1, N_REPLICATES + 1):

        # Every replicate goes into the same folder; the file names
        # carry the replicate number, e.g. ..._explanation_rep1.md.
        replicate_tag = f"rep{replicate}"

        print("\n######################################")
        print(f"REPLICATE {replicate} / {N_REPLICATES}")
        print("######################################")

        for unit in units:

            unit_results = get_unit_results(df, unit)

            if unit_results.empty:
                continue

            if RUN_DECISION:
                run_decision(unit_results, unit, explainability, replicate_tag)

            # -----------------------------------
            # Explanation Agent: how the model with
            # PFI / SHAP makes its decisions
            # -----------------------------------

            if not RUN_EXPLANATION:
                continue

            explanation_package = prepare_explanation_package(
                unit_results, unit, explainability, predictions
            )

            if explanation_package is None:
                print("\nNo PFI / SHAP for this unit, skipping explanation.")
                continue

            grouped_shap = explanation_package["group_shap"]
            features = grouped_shap["columns"]

            for cohort, groups in grouped_shap.items():
                if cohort == "columns":
                    continue

                for outcome, summary in groups.items():
                    print(f"\n{cohort} — {outcome}: {summary['n_records']} records")

                    if summary["n_records"] == 0:
                        continue

                    for feature, signed, absolute in zip(
                        features,
                        summary["mean_shap"],
                        summary["mean_abs_shap"],
                    ):
                        print(
                            f"{feature}: "
                            f"mean SHAP = {signed}, "
                            f"mean absolute SHAP = {absolute}"
                        )

            continue  # Skip the explanation call and report saving below

            print("\n======================================")
            print("RUNNING EXPLANATION AGENT")
            print(format_unit_label(unit))
            print(f"Explained method: {explanation_package['explained_method']}")
            print(f"Explanation model: {MODEL_NAME}")
            print("======================================")

            # e.g. "Converted_qwen3_30b_temp0": target, the LLM that
            # wrote the report, then its temperature.
            name = (
                f"{format_unit_filename(unit)}_"
                f"{get_llm_label(MODEL_NAME)}_"
                f"{get_temperature_label()}"
            )

            print(f"Output name: {name}")

            os.makedirs(OUTPUT_FOLDER, exist_ok=True)

            # Token limits from MODEL_CONFIGS in llm_client.
            # No run_dir: only the final report is saved; the word
            # count and checks are printed to the console.
            _, explanation = explain_unit(
                explanation_package,
                get_explanation_ask(MODEL_NAME),
                MODEL_NAME,
            )

            print("\nEXPLANATION AGENT RESPONSE:")
            print(explanation)

            file_name = f"{name}_explanation_{replicate_tag}.md"

            output_path = os.path.join(OUTPUT_FOLDER, file_name)

            with open(output_path, "w", encoding="utf-8") as file:

                file.write(explanation)

            print("\nSaved explanation to:")
            print(output_path)

    print("\n======================================")
    print("ALL ANALYSIS UNITS COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()
