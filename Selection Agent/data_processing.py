import os

import pandas as pd

# Columns that identify an analysis unit.
# master_results.csv uses Dataset + Timepoint.
# ADNI_model_data.xlsx uses Target.
# Only the ones actually present in the file are used.
POSSIBLE_UNIT_COLUMNS = ["Dataset", "Timepoint", "Target"]


# load the csv/excel data
def load_results(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")

    # before cleaning:
    print("\nbefore cleaning")
    print(df["Method"].unique())

    method_name_map = {
        "RandomForest": "Manual Random Forest",
        "XGBoost": "Manual XGBoost",
        "ExtraTreeClassifier": "TPOT",
    }

    df["Method"] = df["Method"].str.strip()
    df["Method"] = df["Method"].replace(method_name_map)
    # after cleaning:
    print("\nafter cleaning")
    print(df["Method"].unique())
    df["ROC_Gap"] = df["Val_ROC_AUC"] - df["Test_ROC_AUC"]
    df["F1_Gap"] = df["Val_F1"] - df["Test_F1"]

    return df


# ---------------------------------------------
# UNIT HANDLING
# ---------------------------------------------

# Which of the possible unit columns this file actually has.


def get_unit_columns(df):
    return [column for column in POSSIBLE_UNIT_COLUMNS if column in df.columns]


# Every analysis unit in the file, as a list of dicts.
# master_results.csv -> [{"Dataset": ..., "Timepoint": ...}, ...]
# ADNI_model_data.xlsx -> [{"Target": "Converted"}]


def get_units(df):

    unit_columns = get_unit_columns(df)

    if not unit_columns:
        raise ValueError(
            "No unit column found. Expected one of: " + ", ".join(POSSIBLE_UNIT_COLUMNS)
        )

    combinations = df[unit_columns].dropna().drop_duplicates()

    return combinations.to_dict(orient="records")


# each unit is one combination of the unit columns


def get_unit_results(df, unit):

    mask = pd.Series(True, index=df.index)

    for column, value in unit.items():
        mask = mask & (df[column] == value)

    return df[mask].copy()


# Readable label for prompts and printing, e.g.
# "Dataset: Biomarkers Only | Timepoint: 3MO"


def format_unit_label(unit):
    return " | ".join(f"{column}: {value}" for column, value in unit.items())


# Filename-safe version of the unit, e.g.
# "Biomarkers_Only_3MO"


def format_unit_filename(unit):

    parts = []

    for value in unit.values():
        safe_value = str(value).replace(" ", "_").replace("-", "_").replace("/", "_")
        parts.append(safe_value)

    return "_".join(parts)


# Get Top Ranked results for each unit


def get_top_rank_results(unit):
    top_results = unit[unit["Rank"] == 1].copy()
    return top_results


def summarize_methods(unit):
    """Across-fold summaries only. Does not choose a winner."""

    metrics = [
        "Val_ROC_AUC",
        "Test_ROC_AUC",
        "ROC_Gap",
        "Val_F1",
        "Test_F1",
        "F1_Gap",
    ]


def prepare_analysis_package(unit_results, unit):

    # -----------------------------------
    # 1. PRIMARY RESULTS
    # Rank 1 from each method in this unit
    # -----------------------------------

    top_results = get_top_rank_results(unit_results)

    primary_comparison = top_results.to_dict(orient="records")

    # -----------------------------------
    # 2. SUPPORTING RAW RESULTS
    # Ranks 2-10, grouped by method
    # -----------------------------------

    supporting_results_by_method = {}

    for method in unit_results["Method"].unique():

        method_results = unit_results[unit_results["Method"] == method].copy()

        supporting_results = method_results[method_results["Rank"] != 1].copy()

        supporting_results_by_method[method] = supporting_results.to_dict(
            orient="records"
        )

    # -----------------------------------
    # 3. FINAL PACKAGE
    # -----------------------------------

    package = {
        "unit": unit,
        "unit_label": format_unit_label(unit),
        "methods": [str(method) for method in unit_results["Method"].unique()],
        "primary_comparison": primary_comparison,
        "supporting_results_by_method": supporting_results_by_method,
        # "method_summaries":
        #     summarize_methods(unit)
    }

    return package
