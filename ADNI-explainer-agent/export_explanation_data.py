import pandas as pd

from data_processing import (
    OUTCOMES,
    add_outcome_column,
    get_unit_results,
    get_units,
    load_explainability,
    load_predictions,
    load_results,
    match_predictions_sheet,
    num,
    prepare_explanation_package,
)

# Write the <RECORDS> and <GROUP_SHAP> blocks of the explanation prompt
# to one Excel file, laid out like shap_generated_table_with_summary.xlsx
# so the two can be compared by hand. Values are the rounded values the
# LLM sees.

FILE_PATH = "data/ADNI_model_data.xlsx"
OUTPUT_PATH = "data/python_computed_for_prompt.xlsx"
PROFESSOR_PATH = "data/shap_generated_table_with_summary.xlsx"

# Tag order of the "SHAP means by Tag" sheet in the professor's file.
TAG_ORDER = ["TP-CN", "FP-CN", "TN-CN", "FN-CN", "TP-MCI", "FP-MCI", "TN-MCI", "FN-MCI"]


# <RECORDS> as one table, one row per record, with RID and the
# diagnosis labels added back so rows can be matched to the source.
# Built the same way as records_by_group and checked against it.


def records_table(package, explainability, predictions):

    per_sample = explainability["shap_per_sample"]
    labelled, _ = match_predictions_sheet(per_sample, predictions)
    rows = add_outcome_column(labelled)

    columns = package["records_by_group"]["columns"]
    features = columns[1:]

    table = []

    for cohort in ["CN", "MCI"]:
        for outcome in OUTCOMES:

            group = rows[
                (rows["Baseline_Dx"] == cohort) & (rows["Outcome"] == outcome)
            ].sort_values("Probability", ascending=False, kind="mergesort")

            values = [
                [num(record["Probability"])] + [num(record[f]) for f in features]
                for _, record in group.iterrows()
            ]

            if values != package["records_by_group"][cohort][outcome]:
                raise ValueError(f"records of {outcome}-{cohort} differ from the package")

            for (_, record), value in zip(group.iterrows(), values):
                table.append(
                    {
                        "prompt_order": len(table) + 1,
                        "RID": record["RID"],
                        "Y0_actual_diagnosis": record["Baseline_Dx"],
                        "Y1_actual_diagnosis": record["Outcome_Dx"],
                        "Actual": record["Actual"],
                        "Predicted": record["Predicted"],
                        "Tag": f"{outcome}-{cohort}",
                        "raw_probability": record["Probability"],
                        **dict(zip(columns, value)),
                    }
                )

    return pd.DataFrame(table)


# Put the records in the row order of the professor's SHAP_per_sample
# sheet. A RID can appear twice, so rows are matched on RID and
# probability.


def in_professor_order(records):

    professor = pd.read_excel(PROFESSOR_PATH, sheet_name="SHAP_per_sample")

    position = {
        key: i
        for i, key in enumerate(zip(professor["RID"], professor["Probability"].round(6)))
    }

    order = [
        position.get(key)
        for key in zip(records["RID"], records["raw_probability"].round(6))
    ]

    if None in order or len(set(order)) != len(records):
        raise ValueError("records do not match the professor's file one to one")

    return records.assign(_order=order).sort_values("_order").drop(
        columns=["_order", "raw_probability"]
    )


# <GROUP_SHAP> as one table: the mean_shap rows first, one per Tag as
# in the professor's "SHAP means by Tag" sheet, then the mean_abs_shap rows.


def group_table(package):

    group_shap = package["group_shap"]
    table = []

    for key in ["mean_shap", "mean_abs_shap"]:
        for tag in TAG_ORDER:
            outcome, cohort = tag.split("-")
            group = group_shap[cohort][outcome]
            values = group[key] or [None] * len(group_shap["columns"])
            table.append(
                {
                    "Statistic": key,
                    "Tag": tag,
                    "N": group["n_records"],
                    **dict(zip(group_shap["columns"], values)),
                }
            )

    return pd.DataFrame(table)


def main():

    df = load_results(FILE_PATH)
    explainability = load_explainability(FILE_PATH)
    predictions = load_predictions(FILE_PATH)

    unit = get_units(df)[0]
    package = prepare_explanation_package(
        get_unit_results(df, unit), unit, explainability, predictions
    )

    sheets = {
        "RECORDS": in_professor_order(records_table(package, explainability, predictions)),
        "GROUP_SHAP": group_table(package),
    }

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        for name, sheet in sheets.items():
            sheet.to_excel(writer, sheet_name=name, index=False)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
