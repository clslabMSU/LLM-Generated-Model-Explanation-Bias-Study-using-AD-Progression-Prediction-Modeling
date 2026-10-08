import os

import matplotlib.pyplot as plt
import numpy as np

from data_processing import (
    load_results,
    get_units,
    get_unit_results,
    format_unit_label,
    format_unit_filename
)


FILE_PATH = "data/ADNI_model_data.xlsx"


# Preferred order. Only the methods actually present
# in the file are plotted, in this order.
# Any method not listed here is still plotted, at the end.
METHOD_ORDER = [
    # master_results
    "Manual Random Forest",
    "Manual XGBoost",
    "FLAML",
    "FLAML Optimized",
    "TPOT",

    # ADNI_model_data
    "RandomForest",
    "XGBoost",
    "ExtraTreeClassifier"
]


def order_methods(methods):

    present = list(methods)

    ordered = [
        method
        for method in METHOD_ORDER
        if method in present
    ]

    # Anything not in METHOD_ORDER still gets plotted.
    extras = [
        method
        for method in present
        if method not in METHOD_ORDER
    ]

    return ordered + extras


def plot_roc_comparisons(df):

    # We only want the top-ranked result
    # for each method.
    rank1_results = df[
        df["Rank"] == 1
    ].copy()

    # Folder where graphs will be saved
    output_dir = "visualizations"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Loop through each analysis unit
    for unit in get_units(rank1_results):

        current_results = get_unit_results(
            rank1_results,
            unit
        )

        if current_results.empty:
            continue

        # Put the methods in a consistent order.
        current_results["Method"] = (
            current_results["Method"]
            .astype(str)
        )

        method_order = order_methods(
            current_results["Method"].unique()
        )

        current_results = (
            current_results
            .set_index("Method")
            .reindex(method_order)
            .reset_index()
        )

        

        x = np.arange(
            len(current_results)
        )

        bar_width = 0.35

        

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        validation_bars = ax.bar(
            x - bar_width / 2,
            current_results[
                "Val_ROC_AUC"
            ],
            bar_width,
            label="Validation ROC AUC"
        )

        test_bars = ax.bar(
            x + bar_width / 2,
            current_results[
                "Test_ROC_AUC"
            ],
            bar_width,
            label="Test ROC AUC"
        )

        
        ax.bar_label(
            validation_bars,
            fmt="%.4f",
            padding=3,
            fontsize=8
        )

        ax.bar_label(
            test_bars,
            fmt="%.4f",
            padding=3,
            fontsize=8
        )

        
        ax.set_title(
            f"{format_unit_label(unit)}\n"
            "Rank-1 Validation vs Test ROC AUC"
        )

        ax.set_xlabel(
            "Method"
        )

        ax.set_ylabel(
            "ROC AUC"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            current_results["Method"],
            rotation=20,
            ha="right"
        )

        ax.set_ylim(
            0,
            1
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        # Legend at bottom-right of the figure
        fig.legend(
            handles=[
                validation_bars,
                test_bars
            ],
            labels=[
                "Validation ROC AUC",
                "Test ROC AUC"
            ],
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            ncol=2,
            frameon=False
        )


        # Leave space at the bottom for the legend
        fig.tight_layout(
            rect=[0, 0.08, 1, 1]
        )

       

        file_name = (
            f"{format_unit_filename(unit)}_ROC.png"
        )

        file_path = os.path.join(
            output_dir,
            file_name
        )

        fig.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            "Saved:",
            file_path
        )


if __name__ == "__main__":

    print(
        "\nLoading data for visualization..."
    )

    df = load_results(
        FILE_PATH
    )

    print(
        "\nCreating ROC visualizations..."
    )

    plot_roc_comparisons(
        df
    )

    print(
        "\nAll visualizations completed."
    )
