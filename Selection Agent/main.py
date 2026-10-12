import os

from data_processing import (
    load_results,
    get_units,
    get_unit_results,
    get_unit_columns,
    format_unit_label,
    format_unit_filename,
    prepare_analysis_package,
)
from decision_Agent import build_decision_prompt
from llm_client import ask_model


FILE_PATH = "data/ADNI_model_data.xlsx"
NUMBER_OF_RUNS = 5
SEED =  42
TEMPERATURE = 1


AGENTS = (
    ("qwen3:30b", build_decision_prompt, "decision_results_qwen", True, 65536, 16384),
    ("gpt-oss:20b", build_decision_prompt, "decision_results_gpt", "medium", 32768, 8192),
)


def main():
    df = load_results(FILE_PATH)
    df.columns = df.columns.str.strip()

    print("Data got loaded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    unit_columns = get_unit_columns(df)
    units = get_units(df)

    print("\nUnit columns:")
    print(unit_columns)
    print("\nUnits found:")
    for unit in units:
        print(" -", format_unit_label(unit))

    print("\nMethods:")
    print(df["Method"].dropna().unique())

    for unit in units:
        unit_results = get_unit_results(df, unit)
        if unit_results.empty:
            continue

        analysis_package = prepare_analysis_package(unit_results, unit)
        file_name = f"{format_unit_filename(unit)}_decision.txt"

        for model_name, build_prompt, output_folder, think, num_ctx, num_predict in AGENTS:
            # Each run of this model receives exactly the same prompt and data.
            prompt = build_prompt(analysis_package) + ( f"\n\nIn ## Provenance, write: - Decision model: {model_name}")

            for run_number in range(1, NUMBER_OF_RUNS + 1):
                seed = SEED
                run_folder = os.path.join(output_folder, f"run_{run_number:02d}")
                os.makedirs(run_folder, exist_ok=True)
                output_path = os.path.join(run_folder, file_name)

               

                print("\n======================================")
                print(f"UNIT: {format_unit_label(unit)}")
                print(f"MODEL: {model_name}")
                print(f"RUN: {run_number}/{NUMBER_OF_RUNS}; SEED: {seed}")
                print("======================================")

                decision = ask_model(
                    prompt,
                    model_name=model_name,
                    think=think,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                    temperature=TEMPERATURE,
                    seed=seed,
                )

                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(decision)

                print(f"Saved decision to: {output_path}")

    print("\n======================================")
    print("ALL ANALYSIS UNITS COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()
