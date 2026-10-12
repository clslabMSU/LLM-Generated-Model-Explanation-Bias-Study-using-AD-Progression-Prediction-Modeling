"""Standalone Qwen agent for confusion matrices and global SHAP importance.

Install: pip install ollama pandas matplotlib openpyxl
Run: python confusion_matrix_builder.py --decision decision_results_qwen/run_01/Converted_decision.txt --data-root data/result
"""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ollama import chat


# The prompt for the standalone figure agent.
SYSTEM_PROMPT = """You create figures for the method chosen by the decision agent.
Use tools in this order:
1. read_decision. Read the Markdown (or older JSON) decision; identify its
   selected_method, selected_estimator and target. Never choose another winner.
2. list_files starting at "." to find Excel workbooks under the data root.
   Use selected_estimator to identify the model workbook if filenames name the
   estimator rather than the method (for example, ExtraTreeClassifier for TPOT).
   Do not choose a runner-up workbook.
3. Call list_sheets on the candidate workbook. A "prediction folder" inside an
   Excel file is a worksheet/tab. Find its *_Predictions sheet, then inspect_sheet
   to confirm it has ALL_Actual and ALL_Predicted. Inspect other candidates if
   necessary. If several sheets qualify, explain ambiguity and stop.
4. Call build_confusion_matrix with workbook and sheet_name. The tool saves
   three count tables, three titled figures and a combined figure.
   Do not invent or estimate counts.
5. Call build_shap_figure. It saves a global SHAP bar chart only when the SHAP
   rows can be verified as belonging to the selected method. If it is not
   available, say so without substituting another method's SHAP values.
6. Give a brief status for the confusion matrices and SHAP. Do not include
   output file paths or links in your final response.
Actual labels are matrix rows; predicted labels are columns.
The 0 class means No decline and the 1 class means Decline in each panel.
The titles identify Overall, CN→MCI, and MCI→AD.
If a tool reports an error, explain it without inventing results."""


def slug(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def read_model_decision(path):
    
    text = path.read_text(encoding="utf-8-sig").strip()
    if text.startswith("{"):
        decision, _ = json.JSONDecoder().raw_decode(text)
        if not isinstance(decision, dict):
            raise ValueError("Decision JSON must be an object")
    else:
        def field(label):
            match = re.search(rf"^\*\*{re.escape(label)}:\*\*\s*(.+)$",
                              text, flags=re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else ""

        unit = field("Unit")
        match = re.search(r"\bTarget:\s*(.+)$", unit, flags=re.IGNORECASE)
        decision = {"selected_method": field("Selected method"),
                    "selected_estimator": field("Selected estimator"),
                    "target": match.group(1).strip() if match else unit or path.stem}
    if not str(decision.get("selected_method") or "").strip():
        raise ValueError("Decision must contain a selected method")
    return decision


def count_predictions(frame):
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate worksheet column names")
    for name in ("ALL_Actual", "ALL_Predicted"):
        if name not in frame:
            raise ValueError(f"Missing {name} in prediction worksheet")
    actual = frame["ALL_Actual"].fillna("").astype(str).str.strip()
    predicted = frame["ALL_Predicted"].fillna("").astype(str).str.strip()
    if frame.empty or actual.eq("").any() or predicted.eq("").any():
        raise ValueError("Prediction file is empty or has blank labels")
    if not (set(actual) | set(predicted)) <= {"0", "1"}:
        raise ValueError("Expected binary 0/1 values in ALL_Actual and ALL_Predicted")
    matrix = pd.crosstab(actual, predicted).reindex(
        index=["0", "1"], columns=["0", "1"], fill_value=0
    )
    matrix.index.name, matrix.columns.name = "Actual", "Predicted"
    return matrix


PANEL_SPECS = (
    ("overall", "Decline: Overall", ("No decline", "Decline")),
    ("cn_to_mci", "Decline: CN→MCI", ("No decline", "Decline")),
    ("mci_to_ad", "Decline: MCI→AD", ("No decline", "Decline")),
)


def draw_panel(ax, matrix, method, label, diagnosis_labels, show_ylabel=True):
    
    counts = matrix.to_numpy()
    percentages = counts / max(int(counts.sum()), 1) * 100
    color_scale = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    ax.set_title(label, fontsize=12, pad=13)
    ax.text(0.98, 1.02, f"n={counts.sum()}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#455468")
    ax.set_xticks([0, 1], labels=diagnosis_labels)
    ax.set_yticks([0, 1], labels=diagnosis_labels)
    ax.set_xlabel("Predicted")
    if show_ylabel:
        ax.set_ylabel("Actual")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{counts[row, col]}\n({percentages[row, col]:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="semibold",
                    color="white" if percentages[row, col] >= 55 else "#14283b")
    return color_scale


def add_figure_note(fig):
    fig.text(0.5, 0.025,
             "Cells show patients and % of everyone in that matrix. 0 = No decline; 1 = Decline.",
             ha="center", fontsize=10, color="#455468")


def save_combined_plot(matrices, method, output):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for ax, (key, label, diagnosis_labels) in zip(axes, PANEL_SPECS):
        color_scale = draw_panel(ax, matrices[key], method, label, diagnosis_labels,
                                 show_ylabel=key == "overall")
    bar = fig.colorbar(color_scale, ax=axes, shrink=0.66, pad=0.025)
    bar.set_label("% of patients in matrix")
    add_figure_note(fig)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_single_plot(matrix, method, label, diagnosis_labels, output):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    color_scale = draw_panel(ax, matrix, method, label, diagnosis_labels)
    bar = fig.colorbar(color_scale, ax=ax, shrink=0.72, pad=0.04)
    bar.set_label("% of patients in matrix")
    add_figure_note(fig)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalized_method(value):
    """Match the method names in the results workbook to the decision."""
    name = str(value).strip()
    return {"RandomForest": "Manual Random Forest",
            "XGBoost": "Manual XGBoost",
            "ExtraTreeClassifier": "TPOT"}.get(name, name)


def test_cm_counts(value):
    """Extract TN, FP, FN, TP counts from a Test_CM string."""
    pairs = re.findall(r"\b(TN|FP|FN|TP)\s*=\s*(\d+)\b", str(value), re.I)
    counts = {name.upper(): int(number) for name, number in pairs}
    return counts if len(pairs) == 4 and len(counts) == 4 else None


def save_global_shap_figure(shap_workbook, target, method, output_dir):
    
    if not shap_workbook.is_file():
        return {"status": "unavailable", "reason": "SHAP workbook was not found"}
    with pd.ExcelFile(shap_workbook, engine="openpyxl") as book:
        needed = {"SHAP_Test_BestModel", "SHAP_per_sample"}
        if not needed.issubset(book.sheet_names):
            return {"status": "unavailable", "reason": "Global or per-patient SHAP sheet is missing"}
        results = book.parse(book.sheet_names[0])
        global_shap = book.parse("SHAP_Test_BestModel")
        per_sample = book.parse("SHAP_per_sample")

    for frame in (results, global_shap, per_sample):
        frame.columns = frame.columns.astype(str).str.strip()
    if not {"Method", "Test_CM"}.issubset(results.columns):
        return {"status": "unavailable", "reason": "Results sheet lacks Method or Test_CM"}
    if "Target" in results.columns:
        results = results[results["Target"].astype(str).str.strip().str.casefold() == target.casefold()]
    if results.empty or not {"Actual", "Predicted"}.issubset(per_sample.columns):
        return {"status": "unavailable", "reason": "Target rows or per-patient labels are missing"}

    actual = pd.to_numeric(per_sample["Actual"], errors="coerce")
    predicted = pd.to_numeric(per_sample["Predicted"], errors="coerce")
    if per_sample.empty or actual.isna().any() or predicted.isna().any() or not (
        set(actual) | set(predicted)
    ) <= {0, 1}:
        return {"status": "unavailable", "reason": "Per-patient SHAP labels must be binary 0/1"}
    expected = {"TN": int(((actual == 0) & (predicted == 0)).sum()),
                "FP": int(((actual == 0) & (predicted == 1)).sum()),
                "FN": int(((actual == 1) & (predicted == 0)).sum()),
                "TP": int(((actual == 1) & (predicted == 1)).sum())}
    matched = {normalized_method(row["Method"])
               for _, row in results.iterrows() if test_cm_counts(row["Test_CM"]) == expected}
    if len(matched) != 1 or method not in matched:
        return {"status": "unavailable",
                "reason": "SHAP patients do not uniquely match the selected model's Test_CM"}

    if not {"Feature", "SHAP_Mean_Abs"}.issubset(global_shap.columns):
        return {"status": "unavailable", "reason": "Global SHAP lacks Feature or SHAP_Mean_Abs"}
    values = global_shap[["Feature", "SHAP_Mean_Abs"]].dropna(how="all").copy()
    values["Feature"] = values["Feature"].astype(str).str.strip()
    values["SHAP_Mean_Abs"] = pd.to_numeric(values["SHAP_Mean_Abs"], errors="coerce")
    if (values.empty or values["Feature"].eq("").any()
            or values["Feature"].duplicated().any()
            or not values["SHAP_Mean_Abs"].map(math.isfinite).all()
            or (values["SHAP_Mean_Abs"] < 0).any()):
        return {"status": "unavailable", "reason": "Global SHAP feature values are invalid"}
    top = values.nlargest(10, "SHAP_Mean_Abs").sort_values("SHAP_Mean_Abs")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(top) + 1.5)))
    ax.barh(top["Feature"], top["SHAP_Mean_Abs"], color="#5489b3")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"Global SHAP importance — {method}")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{slug(target)}_{slug(method)}_global_shap.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"status": "saved", "selected_method": method}


def tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {},
                       "required": required or []}}}


TOOLS = [
    tool("read_decision", "Read the first Qwen agent's decision file."),
    tool("list_files", "List immediate subfolders and Excel workbooks in a data directory.",
         {"directory": {"type": "string", "description": "Relative to data root; '.' means root"}},
         ["directory"]),
    tool("list_sheets", "List worksheet/tab names inside an Excel workbook.",
         {"workbook": {"type": "string", "description": "XLSX path relative to data root"}},
         ["workbook"]),
    tool("inspect_sheet", "Read worksheet headers to check prediction columns.",
         {"workbook": {"type": "string", "description": "XLSX path relative to data root"},
          "sheet_name": {"type": "string", "description": "Exact sheet name from list_sheets"}},
         ["workbook", "sheet_name"]),
    tool("build_confusion_matrix", "Save three count CSVs, three titled PNGs and one combined PNG.",
         {"workbook": {"type": "string", "description": "XLSX path relative to data root"},
          "sheet_name": {"type": "string", "description": "Exact predictions sheet name"}},
         ["workbook", "sheet_name"]),
    tool("build_shap_figure", "Save a global SHAP bar chart for the selected model when the source can be verified."),
]


def run_agent(decision_path, data_root, output_dir, model,
              shap_workbook=Path("data/ADNI_model_data.xlsx")):
    decision_path, data_root = decision_path.resolve(), data_root.resolve()
    shap_workbook = shap_workbook.resolve()

    def inside_root(relative):
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError("Path must stay inside --data-root")
        return path

    def read_decision(_args):
        return read_model_decision(decision_path)

    def list_files(args):
        directory = inside_root(args["directory"])
        if not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        entries = sorted(directory.iterdir())
        return {"directory": str(directory.relative_to(data_root)),
                "subfolders": [str(p.relative_to(data_root)) for p in entries if p.is_dir()],
                "workbooks": [str(p.relative_to(data_root)) for p in entries
                              if p.is_file() and p.suffix.lower() == ".xlsx"
                              and not p.name.startswith("~$")]}

    def workbook_path(args):
        path = inside_root(args["workbook"])
        if not path.is_file() or path.suffix.lower() != ".xlsx":
            raise ValueError(f"Excel workbook not found: {path}")
        return path

    def list_sheets(args):
        path = workbook_path(args)
        with pd.ExcelFile(path, engine="openpyxl") as book:
            return {"workbook": str(path.relative_to(data_root)), "sheets": book.sheet_names}

    def inspect_sheet(args):
        path = workbook_path(args)
        columns = list(pd.read_excel(path, sheet_name=args["sheet_name"], nrows=0,
                                     engine="openpyxl").columns.astype(str).str.strip())
        return {"workbook": str(path.relative_to(data_root)),
                "sheet_name": args["sheet_name"], "columns": columns,
                "has_prediction_columns": all(c in columns for c in ("ALL_Actual", "ALL_Predicted"))}

    def build_confusion_matrix(args):
        path = workbook_path(args)
        decision = read_model_decision(decision_path)
        method = decision["selected_method"].strip()
        estimator = str(decision.get("selected_estimator") or "").strip()
        sheet = args["sheet_name"]
        expected_sheets = {slug(method) + "predictions"}
        if estimator:
            expected_sheets.add(slug(estimator) + "predictions")
            workbook_names = {slug(method), slug(estimator)}
            if slug(estimator) == "extratreeclassifier":
                workbook_names.add("extratreesclassifier")
            if not any(name in slug(path.stem) for name in workbook_names):
                raise ValueError("Workbook does not match the selected estimator or method")
        if slug(sheet) not in expected_sheets:
            raise ValueError("Sheet is not the selected method's Predictions sheet")
        frame = pd.read_excel(path, sheet_name=sheet, dtype=str, engine="openpyxl")
        if not {"Y0_actual_diagnosis", "Y1_actual_diagnosis"}.issubset(frame.columns):
            raise ValueError("Need Y0_actual_diagnosis and Y1_actual_diagnosis for transition matrices")
        target = str(decision.get("target") or decision_path.stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{slug(target)}_{slug(method)}_confusion_matrix"
        cohorts = {"overall": frame}
        baseline = frame["Y0_actual_diagnosis"].fillna("").astype(str).str.strip().str.upper()
        followup = frame["Y1_actual_diagnosis"].fillna("").astype(str).str.strip().str.upper()
        for diagnosis, next_diagnosis, label in (
            ("CN", "MCI", "cn_to_mci"),
            ("MCI", "DEMENTIA", "mci_to_ad"),
        ):
            group = frame.loc[baseline == diagnosis]
            if group.empty:
                raise ValueError(f"No {diagnosis} baseline rows in the predictions sheet")
            allowed = (followup.loc[group.index] == diagnosis) | (followup.loc[group.index] == next_diagnosis)
            if not allowed.all():
                raise ValueError(f"{diagnosis} baseline includes outcomes besides {diagnosis} and {next_diagnosis}")
            actual = group["ALL_Actual"].fillna("").astype(str).str.strip()
            expected_actual = (followup.loc[group.index] == next_diagnosis).map({True: "1", False: "0"})
            if not actual.equals(expected_actual):
                raise ValueError(f"ALL_Actual does not consistently mean {diagnosis} to {next_diagnosis}")
            cohorts[label] = group
        outputs = {}
        matrices = {}
        panel_by_key = {key: (label, names) for key, label, names in PANEL_SPECS}
        for cohort, rows in cohorts.items():
            matrix = count_predictions(rows)
            matrices[cohort] = matrix
            suffix = "" if cohort == "overall" else f"_{cohort}"
            csv_output = output_dir / f"{stem}{suffix}.csv"
            png_output = output_dir / f"{stem}{suffix}.png"
            matrix.to_csv(csv_output)
            label, diagnosis_labels = panel_by_key[cohort]
            save_single_plot(matrix, method, label, diagnosis_labels, png_output)
            outputs[cohort] = {"rows": int(matrix.to_numpy().sum()),
                               "counts": matrix.to_dict()}
        figure_output = output_dir / f"{stem}_three_panel.png"
        save_combined_plot(matrices, method, figure_output)
        return {"status": "saved", "selected_method": method, "sheet_name": sheet,
                "matrices": outputs}

    def build_shap_figure(_args):
        decision = read_model_decision(decision_path)
        return save_global_shap_figure(
            shap_workbook, str(decision.get("target") or "Converted").strip(),
            decision["selected_method"].strip(), output_dir,
        )

    functions = {"read_decision": read_decision, "list_files": list_files,
                 "list_sheets": list_sheets, "inspect_sheet": inspect_sheet,
                 "build_confusion_matrix": build_confusion_matrix,
                 "build_shap_figure": build_shap_figure}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create confusion matrices and a global SHAP figure. Decision: {decision_path}. Data root: {data_root}."}]
    built = False
    shap_checked = False
    for _ in range(25):
        response = chat(model=model, messages=messages, tools=TOOLS, options={"temperature": 0})
        message = response.message
        messages.append(message)
        if not message.tool_calls:
            if not built or not shap_checked:
                raise RuntimeError(f"Qwen stopped before completing both figure tasks: {message.content}")
            return message.content
        for call in message.tool_calls:
            name = call.function.name
            try:
                result = functions[name](call.function.arguments or {})
                if name == "build_confusion_matrix" and result.get("status") == "saved":
                    built = True
                if name == "build_shap_figure" and result.get("status") in {"saved", "unavailable"}:
                    shap_checked = True
            except (KeyError, OSError, ValueError, pd.errors.ParserError) as exc:
                result = {"error": str(exc)}
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, default=str)})
            print(f"{name}: {json.dumps(result, default=str)}")
    raise RuntimeError("Qwen exceeded 25 tool-use rounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("confusion_matrices"))
    parser.add_argument("--shap-workbook", type=Path,
                        default=Path("data/ADNI_model_data.xlsx"))
    parser.add_argument("--model", default="qwen3:30b")
    args = parser.parse_args()
    if not args.decision.is_file() or not args.data_root.is_dir():
        parser.error("--decision must be a file and --data-root must be a folder")
    print(run_agent(args.decision, args.data_root, args.output_dir, args.model,
                    args.shap_workbook))


if __name__ == "__main__":
    main()
