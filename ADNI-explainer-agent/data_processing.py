import os

import pandas as pd

# Columns that identify an analysis unit.
# master_results.csv uses Dataset + Timepoint.
# ADNI_model_data.xlsx uses Target.
# Only the ones actually present in the file are used.
POSSIBLE_UNIT_COLUMNS = ["Dataset", "Timepoint", "Target"]


# Explainability sheets in ADNI_model_data.xlsx.
# They describe ONE model ("BestModel") and carry no Method
# column, so which method they belong to is worked out from
# the confusion matrix (see identify_explained_method).
EXPLAINABILITY_SHEETS = {
    "pfi": "PFI_Test_BestModel",
    "shap_global": "SHAP_Test_BestModel",
    "shap_per_sample": "SHAP_per_sample",
}


# Columns in SHAP_per_sample that are not features.
SHAP_META_COLUMNS = [
    "RID",
    "Actual",
    "Predicted",
    "Probability",
    "Base_Value",
    "Sum_SHAP",
]


# How many example patients to show per outcome group
# (TP / TN / FP / FN) in the prompt.
SHAP_EXAMPLES_PER_OUTCOME = 2

# How many top drivers to list for each example patient.
SHAP_DRIVERS_PER_EXAMPLE = 5

# Prediction sheets are named "<Method>_Predictions",
# e.g. TPOT_Predictions, RandomForest_Predictions.
PREDICTIONS_SHEET_SUFFIX = "_Predictions"

# Column names differ between result files; map them all to one set.
PREDICTION_COLUMN_MAP = {
    "ALL_RID": "RID",
    "ALL_Actual": "Actual",
    "ALL_Predicted": "Predicted",
    "ALL_Raw_Probability": "Probability",
    "ALL_Raw_Confidence_Score": "Confidence",
    "Y0_actual_diagnosis": "Baseline_Dx",
    "baseline_diagnosis_label": "Baseline_Dx",
    "Y1_actual_diagnosis": "Outcome_Dx",
    "outcome_diagnosis_label": "Outcome_Dx",
}

DIAGNOSIS_LABEL_COLUMNS = ["Baseline_Dx", "Outcome_Dx"]

# Baseline cohorts reported separately, next to the pooled "ALL".
BASELINE_COHORTS = ["CN", "MCI"]

# The model predicts 1 when its probability is at or above this.
DECISION_THRESHOLD = 0.5

# Sheet with the per-fold results. Older files without it
# fall back to their first sheet.
RESULTS_SHEET = "CV_Folds"


# load the csv/excel data
def load_results(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension in (".xlsx", ".xlsm", ".xls"):
        sheet_names = pd.ExcelFile(file_path).sheet_names
        sheet = RESULTS_SHEET if RESULTS_SHEET in sheet_names else 0
        df = pd.read_excel(file_path, sheet_name=sheet)
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


# ---------------------------------------------
# EXPLAINABILITY (PFI / SHAP)
# ---------------------------------------------

# Read the PFI and SHAP sheets, if the file has them.
# Returns None for CSV files or files without those sheets.


def load_explainability(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in (".xlsx", ".xlsm", ".xls"):
        return None

    excel_file = pd.ExcelFile(file_path)

    explainability = {}

    for key, sheet_name in EXPLAINABILITY_SHEETS.items():

        if sheet_name not in excel_file.sheet_names:
            continue

        sheet = excel_file.parse(sheet_name)
        sheet.columns = sheet.columns.str.strip()

        explainability[key] = sheet.dropna(how="all")

    if not explainability:
        return None

    return explainability


# Read every "<Method>_Predictions" sheet, with the columns renamed
# to RID / Actual / Predicted / Probability / Confidence /
# Baseline_Dx / Outcome_Dx. Returns None if there are none.


def load_predictions(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension not in (".xlsx", ".xlsm", ".xls"):
        return None

    excel_file = pd.ExcelFile(file_path)

    predictions = {}

    for sheet_name in excel_file.sheet_names:

        if not sheet_name.endswith(PREDICTIONS_SHEET_SUFFIX):
            continue

        sheet = excel_file.parse(sheet_name)
        sheet.columns = sheet.columns.str.strip()
        sheet = sheet.rename(columns=PREDICTION_COLUMN_MAP).dropna(how="all")

        method = sheet_name[: -len(PREDICTIONS_SHEET_SUFFIX)]
        predictions[method] = sheet

    return predictions or None


# The SHAP sheets say "BestModel" but not which method that is.
# The test-set confusion matrix is unique enough to find it:
# rebuild it from SHAP_per_sample and look it up in Test_CM.


def identify_explained_method(df, explainability):

    per_sample = explainability.get("shap_per_sample")

    if per_sample is None:
        return None

    if "Test_CM" not in df.columns:
        return None

    actual = per_sample["Actual"]
    predicted = per_sample["Predicted"]

    counts = {
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
    }

    wanted = (
        f"(TN={counts['TN']},FP={counts['FP']}," f"FN={counts['FN']},TP={counts['TP']})"
    )

    stored = df["Test_CM"].astype(str).str.replace(" ", "", regex=False)

    matches = df[stored == wanted]["Method"].unique()

    # Only trust it when every matching row is the same method.
    if len(matches) == 1:
        return str(matches[0])

    return None


# Which outcome group a patient falls into.


def get_outcome_group(actual, predicted):

    if actual == 1 and predicted == 1:
        return "TP"

    if actual == 0 and predicted == 0:
        return "TN"

    if actual == 0 and predicted == 1:
        return "FP"

    return "FN"


# SHAP_per_sample is 106 patients x 21 features, which is too
# much raw detail for the prompt. Condensed into overall counts
# plus a few example patients with their strongest drivers.


def summarize_shap_per_sample(per_sample):

    feature_columns = [
        column for column in per_sample.columns if column not in SHAP_META_COLUMNS
    ]

    rows = per_sample.copy()

    rows["Outcome"] = [
        get_outcome_group(actual, predicted)
        for actual, predicted in zip(rows["Actual"], rows["Predicted"])
    ]

    # Most confident calls first.
    rows["Confidence"] = (rows["Probability"] - 0.5).abs()

    correct = int((rows["Actual"] == rows["Predicted"]).sum())

    examples = []

    for outcome in ["TP", "TN", "FP", "FN"]:

        group = rows[rows["Outcome"] == outcome]

        group = group.sort_values("Confidence", ascending=False)

        for _, patient in group.head(SHAP_EXAMPLES_PER_OUTCOME).iterrows():

            drivers = (
                patient[feature_columns]
                .astype(float)
                .abs()
                .sort_values(ascending=False)
                .head(SHAP_DRIVERS_PER_EXAMPLE)
                .index
            )

            examples.append(
                {
                    "RID": int(patient["RID"]),
                    "outcome": outcome,
                    "actual": int(patient["Actual"]),
                    "predicted": int(patient["Predicted"]),
                    "probability": round(float(patient["Probability"]), 4),
                    "top_drivers": {
                        feature: round(float(patient[feature]), 4)
                        for feature in drivers
                    },
                }
            )

    return {
        "n_patients": int(len(rows)),
        "n_correct": correct,
        "n_features": len(feature_columns),
        "base_value": round(float(rows["Base_Value"].iloc[0]), 4),
        "outcome_counts": {
            outcome: int(count)
            for outcome, count in rows["Outcome"].value_counts().items()
        },
        "example_patients": examples,
    }


# Everything the prompt needs about feature importance,
# already tied to the method it belongs to.


def prepare_explainability_package(df, explainability):

    if not explainability:
        return None

    package = {"explained_method": identify_explained_method(df, explainability)}

    if "pfi" in explainability:
        package["permutation_importance"] = explainability["pfi"].to_dict(
            orient="records"
        )

    if "shap_global" in explainability:
        package["shap_global"] = explainability["shap_global"].to_dict(orient="records")

    if "shap_per_sample" in explainability:
        package["shap_per_sample_summary"] = summarize_shap_per_sample(
            explainability["shap_per_sample"]
        )

    return package


# Get Top Ranked results for each unit


def get_best_fold(unit):
    top_results = unit[unit["Rank"] == 1].copy()
    return top_results


def prepare_analysis_package(unit_results, unit, explainability=None):

    # -----------------------------------
    # 1. PRIMARY RESULTS
    # Rank 1 from each method in this unit
    # -----------------------------------

    top_results = get_best_fold(unit_results)

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
        # PFI / SHAP for the best model. None when the file
        # has no explainability sheets.
        "explainability": prepare_explainability_package(unit_results, explainability),
    }

    return package


# ---------------------------------------------
# EXPLANATION PACKAGE (for explanation_agent)
# ---------------------------------------------
#
# The Explanation Agent reads and analyses the data itself. Python
# only reads the sheets, labels every test record with its cohort
# (CN / MCI) and outcome group (TP / TN / FP / FN) and counts the
# confusion matrices. Nothing in the sources is changed: missing
# values stay missing and duplicated IDs stay separate records.
# The consistency checks are printed for the user, not sent to the LLM.

OUTCOMES = ["TP", "TN", "FP", "FN"]

# Observed transition -> Actual label, per the study definition.
TRANSITION_LABELS = {
    ("CN", "MCI"): 1,
    ("CN", "CN"): 0,
    ("MCI", "Dementia"): 1,
    ("MCI", "MCI"): 0,
}

# Absolute tolerance for numerical consistency checks.
CHECK_TOLERANCE = 1e-6


def get_shap_feature_columns(per_sample):
    excluded = SHAP_META_COLUMNS + DIAGNOSIS_LABEL_COLUMNS
    return [column for column in per_sample.columns if column not in excluded]


def add_outcome_column(per_sample):

    rows = per_sample.copy()

    rows["Outcome"] = [
        get_outcome_group(actual, predicted)
        for actual, predicted in zip(rows["Actual"], rows["Predicted"])
    ]

    return rows


# Round for the prompt. Missing stays None (never 0), and a tiny
# non-zero value is not rounded to 0.0, which would hide its sign.


def num(value, digits=3):

    if value is None or pd.isna(value):
        return None

    value = float(value)
    rounded = round(value, digits)

    if rounded == 0 and value != 0:
        return float(f"{value:.2g}")

    return rounded


def numeric(frame):
    return frame.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------
# DIAGNOSIS LABELS AND COHORTS (from <Method>_Predictions)
# ---------------------------------------------

# Find the Predictions sheet that belongs to SHAP_per_sample.
# A RID can appear more than once, so RID alone is not a join key:
# rows are joined by position, and a sheet is accepted only when
# RID, Actual, Predicted and Probability all agree row by row.
# Returns (labelled rows, join information).


def match_predictions_sheet(per_sample, predictions):

    info = {
        "sheet": None,
        "join_key": (
            "row position, accepted only when RID, Actual, Predicted and "
            "Probability agree on every row"
        ),
        "rejected_sheets": {},
    }

    if not predictions:
        info["reason"] = "no <Method>_Predictions sheets in the file"
        return per_sample, info

    for name, sheet in predictions.items():

        missing = [
            column
            for column in ["RID", "Actual", "Predicted", "Probability"]
            + DIAGNOSIS_LABEL_COLUMNS
            if column not in sheet.columns
        ]

        if missing:
            info["rejected_sheets"][name] = "missing columns: " + ", ".join(missing)
            continue

        if len(sheet) != len(per_sample):
            info["rejected_sheets"][
                name
            ] = f"{len(sheet)} rows, SHAP_per_sample has {len(per_sample)}"
            continue

        mismatches = {}

        for column in ["RID", "Actual", "Predicted"]:
            unequal = sheet[column].to_numpy() != per_sample[column].to_numpy()
            if unequal.any():
                mismatches[column] = int(unequal.sum())

        gap = abs(
            sheet["Probability"].to_numpy(dtype=float)
            - per_sample["Probability"].to_numpy(dtype=float)
        )

        if (gap > CHECK_TOLERANCE).any():
            mismatches["Probability"] = int((gap > CHECK_TOLERANCE).sum())

        if mismatches:
            info["rejected_sheets"][name] = "rows differ: " + ", ".join(
                f"{column} on {count} rows" for column, count in mismatches.items()
            )
            continue

        labelled = per_sample.copy()

        for column in DIAGNOSIS_LABEL_COLUMNS:
            labelled[column] = sheet[column].to_numpy()

        info["sheet"] = name
        return labelled, info

    info["reason"] = "no Predictions sheet matched SHAP_per_sample row by row"
    return per_sample, info


def has_diagnosis_labels(rows):
    return set(DIAGNOSIS_LABEL_COLUMNS) <= set(rows.columns)


# The cohorts analysed separately: {"CN": ..., "MCI": ...} when there
# are diagnosis labels (a cohort may be empty), otherwise {"ALL": ...}.


def analysis_cohorts(rows):

    if not has_diagnosis_labels(rows):
        return {"ALL": rows}

    return {cohort: rows[rows["Baseline_Dx"] == cohort] for cohort in BASELINE_COHORTS}


# Pooled "ALL" first, then each cohort.


def split_by_cohort(rows):

    if not has_diagnosis_labels(rows):
        return {"ALL": rows}

    return {"ALL": rows, **analysis_cohorts(rows)}


# ---------------------------------------------
# DATA FOR THE PROMPT
# ---------------------------------------------

# Number of records in each outcome group, for all records (ALL)
# and for each cohort.


def confusion_matrices(labelled):

    rows = add_outcome_column(labelled)

    return {
        cohort: {
            "n_records": int(len(group)),
            **{outcome: int((group["Outcome"] == outcome).sum()) for outcome in OUTCOMES},
        }
        for cohort, group in split_by_cohort(rows).items()
    }


# The PFI and global SHAP sheets side by side, one row per feature,
# with the values as stored (rounded). No ranks or selections.


def global_importance_table(explainability):

    sheets = [
        explainability[key].drop(columns="Group", errors="ignore")
        for key in ["pfi", "shap_global"]
        if explainability.get(key) is not None
    ]

    if not sheets:
        return None

    merged = sheets[0]

    for sheet in sheets[1:]:
        merged = merged.merge(sheet, on="Feature", how="outer", sort=False)

    return [
        {
            "feature": row["Feature"],
            **{column: num(row[column]) for column in merged.columns if column != "Feature"},
        }
        for _, row in merged.iterrows()
    ]


# Every test record, grouped by cohort and outcome group, highest
# probability first: its probability and the signed SHAP value of every
# feature, in the order of "columns". No participant IDs.


def records_by_group(labelled, feature_columns):

    rows = add_outcome_column(labelled)

    groups = {}

    for cohort, cohort_rows in analysis_cohorts(rows).items():

        groups[cohort] = {}

        for outcome in OUTCOMES:

            group = cohort_rows[cohort_rows["Outcome"] == outcome].sort_values(
                "Probability", ascending=False, kind="mergesort"
            )

            groups[cohort][outcome] = [
                [num(record["Probability"])] + [num(record[f]) for f in feature_columns]
                for _, record in group.iterrows()
            ]

    return {"columns": ["probability"] + feature_columns, **groups}


# Group-level SHAP per cohort and outcome group: the mean signed SHAP
# (direction) and the mean absolute SHAP (magnitude) of every feature,
# in the order of "columns" as in the source sheet, not ranked. An empty
# group has n_records 0 and null means.


def group_shap_means(labelled, feature_columns):

    rows = add_outcome_column(labelled)

    groups = {}

    for cohort, cohort_rows in analysis_cohorts(rows).items():

        groups[cohort] = {}

        for outcome in OUTCOMES:

            shap = numeric(cohort_rows.loc[cohort_rows["Outcome"] == outcome, feature_columns])

            groups[cohort][outcome] = {
                "n_records": int(len(shap)),
                "mean_shap": [num(v) for v in shap.mean()] if len(shap) else None,
                "mean_abs_shap": [num(v) for v in shap.abs().mean()] if len(shap) else None,
            }
    print(groups)
    return {"columns": feature_columns, **groups}


def check_shap_reconstruction(per_sample, feature_columns):

    needed = ["Base_Value", "Probability"]

    if any(column not in per_sample.columns for column in needed):
        return None, "not run: Base_Value or Probability column missing"

    shap = numeric(per_sample[feature_columns])

    if shap.isna().any().any() or per_sample[needed].isna().any().any():
        return None, "not run: some SHAP contributions or base values are missing"

    gap = (per_sample["Base_Value"] + shap.sum(axis=1) - per_sample["Probability"]).abs()

    return float(gap.max()), f"largest gap {float(gap.max()):.2e}"


def check_result(name, status, detail):
    return {"check": name, "status": status, "detail": detail}


def pass_or_fail(passed):
    return "pass" if passed else "FAIL"


# Consistency checks on the source sheets and on the grouping.
# Failures are reported, never repaired.


def run_data_checks(unit_results, explainability, per_sample, labelled, join_info, confusion):

    checks = []
    feature_columns = get_shap_feature_columns(per_sample)

    # Required columns and missing values.
    missing_columns = [
        column for column in SHAP_META_COLUMNS if column not in per_sample.columns
    ]
    checks.append(
        check_result(
            "SHAP_per_sample columns",
            pass_or_fail(not missing_columns),
            (
                "all present"
                if not missing_columns
                else "missing: " + ", ".join(missing_columns)
            ),
        )
    )

    n_missing = int(per_sample[feature_columns].isna().sum().sum())
    meta_missing = int(
        per_sample[[c for c in SHAP_META_COLUMNS if c in per_sample.columns]]
        .isna()
        .sum()
        .sum()
    )
    checks.append(
        check_result(
            "missing values",
            "pass" if n_missing == meta_missing == 0 else "warning",
            f"{n_missing} missing SHAP values and {meta_missing} missing identifier, "
            f"label or probability values; missing values are kept missing, never set to 0",
        )
    )

    # Labels and threshold.
    binary = (
        per_sample["Actual"].isin([0, 1]).all()
        and per_sample["Predicted"].isin([0, 1]).all()
    )
    checks.append(
        check_result(
            "binary Actual and Predicted",
            pass_or_fail(binary),
            "values are 0 or 1" if binary else "non-binary values found",
        )
    )

    expected = (per_sample["Probability"] >= DECISION_THRESHOLD).astype(int)
    wrong = int((expected != per_sample["Predicted"]).sum())
    checks.append(
        check_result(
            "predictions match probability and threshold",
            pass_or_fail(wrong == 0),
            f"{wrong} of {len(per_sample)} records disagree with Probability >= {DECISION_THRESHOLD}",
        )
    )

    # Duplicate IDs and the join with the Predictions sheet.
    duplicated = per_sample["RID"][per_sample["RID"].duplicated(keep=False)]
    checks.append(
        check_result(
            "duplicate participant IDs",
            "pass" if duplicated.empty else "warning",
            (
                "every RID appears once"
                if duplicated.empty
                else f"{duplicated.nunique()} RID(s) appear more than once "
                f"({len(duplicated)} records): {sorted(int(r) for r in duplicated.unique())}. "
                f"Records are kept as separate test records, not removed; "
                f"{per_sample['RID'].nunique()} unique RIDs in {len(per_sample)} records"
            ),
        )
    )

    if join_info["sheet"]:
        detail = (
            f"labels from sheet {join_info['sheet']}; join key: {join_info['join_key']}"
        )
        status = "pass"
    else:
        detail = join_info.get("reason", "no match") + (
            f"; rejected: {join_info['rejected_sheets']}"
            if join_info["rejected_sheets"]
            else ""
        )
        status = "warning"
    checks.append(
        check_result("Predictions sheet joined to SHAP_per_sample", status, detail)
    )

    # Cohort / outcome labels against the study definition.
    if has_diagnosis_labels(labelled):
        pairs = list(
            zip(labelled["Baseline_Dx"], labelled["Outcome_Dx"], labelled["Actual"])
        )
        unknown = sorted(
            {f"{b} -> {o}" for b, o, _ in pairs if (b, o) not in TRANSITION_LABELS}
        )
        wrong_label = sum(
            1
            for b, o, a in pairs
            if (b, o) in TRANSITION_LABELS and TRANSITION_LABELS[(b, o)] != a
        )
        checks.append(
            check_result(
                "transitions match Actual labels",
                pass_or_fail(not unknown and wrong_label == 0),
                f"{wrong_label} records whose Actual label contradicts their transition; "
                f"transitions outside the study definition: {unknown or 'none'}",
            )
        )
    else:
        checks.append(
            check_result(
                "transitions match Actual labels",
                "not run",
                "no diagnosis labels attached",
            )
        )

    # Every record falls in exactly one cohort.
    cohorts = [c for c in confusion if c != "ALL"]

    if cohorts:
        problems = [
            f"{key}: CN + MCI does not equal ALL"
            for key in ["n_records"] + OUTCOMES
            if sum(confusion[c][key] for c in cohorts) != confusion["ALL"][key]
        ]
        checks.append(
            check_result(
                "cohort counts add up to ALL",
                pass_or_fail(not problems),
                "every record is in exactly one cohort" if not problems else "; ".join(problems),
            )
        )

    # Explained model vs the stored test confusion matrix.
    method = identify_explained_method(unit_results, explainability)
    checks.append(
        check_result(
            "SHAP sheets matched to one method",
            "pass" if method else "warning",
            (
                f"test confusion matrix of SHAP_per_sample matches Test_CM of {method} only"
                if method
                else "no unique method in CV_Folds has this test confusion matrix"
            ),
        )
    )

    # SHAP arithmetic.
    shap = numeric(per_sample[feature_columns])

    if "Sum_SHAP" in per_sample.columns and not shap.isna().any().any():
        gap = float((shap.sum(axis=1) - per_sample["Sum_SHAP"]).abs().max())
        checks.append(
            check_result(
                "Sum_SHAP equals the sum of feature SHAP values",
                pass_or_fail(gap < CHECK_TOLERANCE),
                f"largest gap {gap:.2e}",
            )
        )
    else:
        checks.append(
            check_result(
                "Sum_SHAP equals the sum of feature SHAP values",
                "not run",
                "Sum_SHAP missing or some SHAP values missing",
            )
        )

    gap, detail = check_shap_reconstruction(per_sample, feature_columns)
    checks.append(
        check_result(
            "SHAP reconstruction: base value + SHAP = probability",
            (
                "not run"
                if gap is None
                else ("pass" if gap < CHECK_TOLERANCE else "warning")
            ),
            detail,
        )
    )

    if "Base_Value" in per_sample.columns:
        base = per_sample["Base_Value"].dropna()
        spread = float(base.max() - base.min()) if not base.empty else float("nan")
        checks.append(
            check_result(
                "single base value",
                pass_or_fail(spread < CHECK_TOLERANCE),
                f"base values differ by at most {spread:.2e} across records",
            )
        )

    # Global sheets against the per-record SHAP.
    shap_global = explainability.get("shap_global")
    pfi = explainability.get("pfi")

    if shap_global is not None:
        stored = shap_global.set_index("Feature")["SHAP_Mean_Abs"]
        common = [f for f in feature_columns if f in stored.index]
        gap = (
            float((shap[common].abs().mean() - stored.reindex(common)).abs().max())
            if common
            else float("nan")
        )
        checks.append(
            check_result(
                "global SHAP equals mean absolute per-record SHAP",
                pass_or_fail(bool(common) and gap < CHECK_TOLERANCE),
                f"largest gap {gap:.2e} over {len(common)} features",
            )
        )

    sets = {"SHAP_per_sample": set(feature_columns)}
    if pfi is not None:
        sets["PFI sheet"] = set(pfi["Feature"])
    if shap_global is not None:
        sets["global SHAP sheet"] = set(shap_global["Feature"])
    same = all(value == sets["SHAP_per_sample"] for value in sets.values())
    checks.append(
        check_result(
            "same features in every sheet",
            pass_or_fail(same),
            (
                f"{len(feature_columns)} features in each of {', '.join(sets)}"
                if same
                else "; ".join(
                    f"{name}: {len(value)} features" for name, value in sets.items()
                )
            ),
        )
    )

    if pfi is not None and "Group" in pfi.columns:
        groups = sorted(str(g) for g in pfi["Group"].dropna().unique())
        checks.append(
            check_result(
                "PFI scope",
                "pass" if groups == ["ALL"] else "warning",
                f"PFI Group values: {groups}; PFI is pooled, not cohort-specific",
            )
        )

    return checks


# ---------------------------------------------
# PACKAGE
# ---------------------------------------------

# The estimator behind a method, e.g. "ExtraTreeClassifier" for
# TPOT, from the Estimator column of its Rank 1 row. None when
# the method or the column is missing.


def get_method_estimator(unit_results, method):

    if method is None or "Estimator" not in unit_results.columns:
        return None

    rows = get_best_fold(unit_results[unit_results["Method"] == method])

    if rows.empty:
        return None

    return str(rows["Estimator"].iloc[0])


# Every test-set metric of the explained model's best fold (its Rank 1
# row in CV_Folds), as stored: all "Test_" columns. None when the
# method or its Rank 1 row is missing.


def get_test_metrics(unit_results, method):

    if method is None:
        return None

    rows = get_best_fold(unit_results[unit_results["Method"] == method])

    if rows.empty:
        return None

    row = rows.iloc[0]

    return {
        "rank": int(row["Rank"]),
        "fold": int(row["Fold_Num"]) if "Fold_Num" in row.index else None,
        **{
            column: str(row[column]) if column == "Test_CM" else num(row[column])
            for column in unit_results.columns
            if column.startswith("Test_")
        },
    }


# Everything the explanation agent needs, tied to the method the
# PFI / SHAP sheets belong to. None when the file has no
# explainability sheets. predictions (from load_predictions) adds
# the CN / MCI cohort labels when available.


def prepare_explanation_package(unit_results, unit, explainability, predictions=None):

    if not explainability:
        return None

    explained_method = identify_explained_method(unit_results, explainability)

    package = {
        "unit": unit,
        "unit_label": format_unit_label(unit),
        "explained_method": explained_method,
        "explained_estimator": get_method_estimator(unit_results, explained_method),
        "test_metrics": get_test_metrics(unit_results, explained_method),
        "global_importance": global_importance_table(explainability),
    }

    per_sample = explainability.get("shap_per_sample")

    if per_sample is None:
        return package

    labelled, join_info = match_predictions_sheet(per_sample, predictions)

    feature_columns = get_shap_feature_columns(per_sample)

    package["features"] = feature_columns
    package["base_value"] = (
        num(per_sample["Base_Value"].iloc[0])
        if "Base_Value" in per_sample.columns
        else None
    )
    package["n_patients"] = int(len(per_sample))
    package["n_unique_rids"] = int(per_sample["RID"].nunique())
    package["n_correct"] = int((per_sample["Actual"] == per_sample["Predicted"]).sum())
    package["has_cohorts"] = has_diagnosis_labels(labelled)
    package["predictions_join"] = join_info
    package["confusion_matrix"] = confusion_matrices(labelled)
    package["records_by_group"] = records_by_group(labelled, feature_columns)
    package["group_shap"] = group_shap_means(labelled, feature_columns)
    package["data_checks"] = run_data_checks(
        unit_results,
        explainability,
        per_sample,
        labelled,
        join_info,
        package["confusion_matrix"],
    )

    return package
