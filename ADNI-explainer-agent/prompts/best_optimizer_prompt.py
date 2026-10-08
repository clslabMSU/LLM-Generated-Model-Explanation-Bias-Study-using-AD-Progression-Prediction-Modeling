# Prompt text for the Decision Agent (decision_agent.py).
# Only text lives here; decision_agent.py prepares the values.
#
# Both texts are copied word for word from the team's code in
# Nour/Adni Conference/ADNI-explainer-agent, so both code bases send
# the same prompts. Re-copy them when the team changes theirs:
#   get_qwen_decision_prompt -> decision_Agent_Qwen.py (qwen3:30b)
#   get_gpt_decision_prompt  -> decision_Agent.py      (gpt-oss:20b)


def get_qwen_decision_prompt(
    unit_lines,
    unit_label,
    method_count,
    method_lines,
    max_folds,
    primary_results_json,
    supporting_results_json,
    explainability_section="",
    feature_explanation_rule="",
):
    prompt = f"""
You are analyzing ExplainTBI model-comparison results.

Python prepared the tables. It did not choose a winner.
Your job is to look at this unit and reason about which method
is the best model. You may notice a pattern that is not obvious
from Rank 1 alone.

Copy numbers exactly. Do not invent values. Do not compute new
averages. Do not round with ~.

- Rank-1 numbers come from <PRIMARY_RESULTS>
- Remaining-fold numbers come from <SUPPORTING_RESULTS>
- A Rank-1 gap is not an average. An average is not a Rank-1 gap.
- final_reasoning must say the same thing


==================================================
THIS UNIT
==================================================

{unit_lines}

{method_count} methods were run on the same data:

{method_lines}


==================================================
HOW THE EXPERIMENT WAS RUN
==================================================

First, patients were split into two separate sets:
- 80% formed the TRAINING set.
- 20% formed the held-out TEST SET. 

Inside the 80% training set, we used {max_folds}-fold
cross-validation. In each fold,
90% of that training data was used to train the model and the
remaining 10% was used for VALIDATION.

- Val_* = the model's performance score on the 10% validation portion of the
  training set.
- Test_* = the model's performance score on the separate,
  unseen 20% test set.

The best fold was selected by highest Val_roc_auc and assigned Rank 1.
Focus on each method's Rank-1 metrics when comparing methods and ignore the remaining fold nums(rank2-10)results .






==================================================
METRICS TO DECIDE OPTIMAL MODEL
==================================================

Reason mainly from:

- Val_ROC_AUC
- Test_ROC_AUC
- ROC_Gap = Val_ROC_AUC - Test_ROC_AUC
- Val_F1
- Test_F1
- F1_Gap  = Val_F1 - Test_F1
- Val_Recall/Sensitivity and Test_Recall/Sensitivity
- Val_Specificity and Test_Specificity
- Val_Precision and Test_Precision

DO NOT REPLACE ANY OF THE METRICS. USE THE EXACT VALUES OF THE PROVIDED TABLES. 

Positive gap: validation > test.
Negative gap: test > validation.
The same reading applies to F1_Gap.

Our goal is to select the model that performs the best taking into account  metrics in order:
ROC_AUC, F1, sensitivity, specificity, Precision, ROC_Gap, F1_Gap.

We want a model that performs very well on both validation and test set. It is helpful for the gap to be minimal as that demonstrates robustness.
Hence why we provide the ROC_Gap and F1_Gap values.
Think of this as a multi-optimization problem taking into account optimizing the varied metrics. The test set is a hold out set so probably the most important factor in determining performance.
However, a model that performs very well on a test set but not on the validation set is questionable. It might not generalize well. 


Explain your reasoning for the most optimal model selected.
==================================================
RANK 1
==================================================

<PRIMARY_RESULTS>
{primary_results_json}
</PRIMARY_RESULTS>


==================================================
RANKS 2-{max_folds}
==================================================

<SUPPORTING_RESULTS>
{supporting_results_json}
</SUPPORTING_RESULTS>
{explainability_section}

==================================================
TASK
==================================================

Using the context and all of the results, reason about the best
method for this unit. Choose one method and a different runner-up.

Return only a concise Markdown document. No JSON, repeated rationale,
or separate presentation_summary. Replace bracketed instructions with
actual content. Keep the headings and the exact selection fields.

# Best model for {unit_label}

*ExplainADNI model-selection decision*

## Answer

**Unit:** {unit_label}
**Selected method:** [exact method name]
**Runner-up method:** [different exact method name]
**First primary criterion metric:** [one exact supplied metric column name]
**Second primary criterion metric:** [a different exact supplied metric column name]

[For the two criteria, name the metrics that actually carried the most weight
in this decision]

**Confidence:** [high, moderate, or low]

[One or two sentences: selection, strongest evidence, and key trade-off.]

## Evidence

*Rank-1 values copied exactly from <PRIMARY_RESULTS>; no new averages.*

[One two-row Markdown table: selected method and runner-up. Include
Method, Estimator, Rank, Fold_Num, Val_ROC_AUC, Test_ROC_AUC,
ROC_Gap, Val_F1, Test_F1, F1_Gap, Test_Precision,
Test_Recall/Sensitivity, and Test_Specificity. Use these exact
column names and source values. Mention Val_Precision,
Val_Recall/Sensitivity, or Val_Specificity only if they materially
change the comparison. Omit fields absent from the supplied rows.]



## Interpretation

[Two to four sentences comparing the methods using exact Rank-1
values and meaningful trade-offs. .]

{feature_explanation_rule}

## Limitations

- [One or two limitations supported by the supplied evidence.
   Do not claim external validation, clinical utility, or causation.]

## Evaluation

**Best method chosen:** [repeat Selected method exactly]
**Runner-up method chosen:** [repeat Runner-up method exactly]
**First primary criterion metric:** [one exact supplied metric column name]
**Second primary criterion metric:** [a different exact supplied metric column name]

[For the two criteria, name the metrics that actually carried the most weight
in this decision]

## Provenance

- Unit: {unit_label}
- Decision model: [model name if supplied by the application;
  otherwise "not recorded in the input"]
- Selection source: <PRIMARY_RESULTS>; each method's Rank 1 was
  chosen by Val_ROC_AUC within the training pool
- Selected row: [exact Method, Estimator, Rank, and Fold_Num]
- Runner-up row: [exact Method, Estimator, Rank, and Fold_Num]
- Supporting source: <SUPPORTING_RESULTS> [name only the folds used]


Check every stated number against its source row. Do not invent
figures, paths, timestamps, costs, audits, or missing metrics.
"""
    return prompt


def get_gpt_decision_prompt(
    unit_lines,
    unit_label,
    method_count,
    method_lines,
    max_folds,
    primary_results_json,
    supporting_results_json,
    explainability_section="",
    feature_explanation_rule="",
):
    prompt = f"""
You are analyzing ExplainTBI model-comparison results.

Python prepared the tables. It did not choose a winner.
Your job is to look at this unit and reason about which method
is the best model. You may notice a pattern that is not obvious
from Rank 1 alone.

Copy numbers exactly. Do not invent values. Do not compute new
averages. Do not round with ~.

- Rank-1 numbers come from <PRIMARY_RESULTS>
- Remaining-fold numbers come from <SUPPORTING_RESULTS>
- A Rank-1 gap is not an average. An average is not a Rank-1 gap.
- final_reasoning must say the same thing


==================================================
THIS UNIT
==================================================

{unit_lines}

{method_count} methods were run on the same data:

{method_lines}


==================================================
HOW THE EXPERIMENT WAS RUN
==================================================

First, patients were split into two separate sets:
- 80% formed the TRAINING set.
- 20% formed the held-out TEST SET. 

Inside the 80% training set, we used {max_folds}-fold
cross-validation. In each fold,
90% of that training data was used to train the model and the
remaining 10% was used for VALIDATION.

- Val_* = the model's performance score on the 10% validation portion of the
  training set.
- Test_* = the model's performance score on the separate,
  unseen 20% test set.

The best fold was selected by highest Val_roc_auc and assigned Rank 1.
Focus on each method's Rank-1 metrics when comparing methods and ignore the remaining fold nums(rank2-10)results.




==================================================
METRICS TO DECIDE OPTIMAL MODEL
==================================================

Reason mainly from:

- Val_ROC_AUC
- Test_ROC_AUC
- ROC_Gap = Val_ROC_AUC - Test_ROC_AUC
- Val_F1
- Test_F1
- F1_Gap  = Val_F1 - Test_F1
- Val_Recall/Sensitivity and Test_Recall/Sensitivity
- Val_Specificity and Test_Specificity
- Val_Precision and Test_Precision

DO NOT REPLACE ANY OF THE METRICS. USE THE EXACT VALUES OF THE PROVIDED TABLES. 

Positive gap: validation > test.
Negative gap: test > validation.
The same reading applies to F1_Gap.

Our goal is to select the model that performs the best taking into account  metrics in order:
ROC_AUC, F1, sensitivity, specificity, Precison, ROC_Gap, F1_Gap.

We want a model that performs very well on both validation and test set. It is helpful for the gap to be minimal as that demonstrates robustness.
Hence why we provide the ROC_Gap and F1_Gap values.
Think of this as a multi-optimization problem taking into account optimizing the varied metrics. The test set is a hold out set so probably the most important factor in determining performance.
However, a model that performs very well on a test set but not on the validation set is questionable. It might not generalize well. 


Explain your reasoning for the most optimal model selected.
==================================================
RANK 1
==================================================

<PRIMARY_RESULTS>
{primary_results_json}
</PRIMARY_RESULTS>


==================================================
RANKS 2-{max_folds}
==================================================

<SUPPORTING_RESULTS>
{supporting_results_json}
</SUPPORTING_RESULTS>
{explainability_section}

==================================================
TASK
==================================================

Using the context and all of the results, reason about the best
method for this unit. Choose one method and a different runner-up.

Return only a concise Markdown document. No JSON, repeated rationale,
or separate presentation_summary. Replace bracketed instructions with
actual content. Keep the headings and the exact selection fields.

# Best model for {unit_label}

*ExplainADNI model-selection decision*

## Answer

**Unit:** {unit_label}
**Selected method:** [exact method name]
**Runner-up method:** [different exact method name]
**First primary criterion metric:** [one exact supplied metric column name]
**Second primary criterion metric:** [a different exact supplied metric column name]

[For the two criteria, name the metrics that actually carried the most weight
in this decision]

**Confidence:** [high, moderate, or low]

[One or two sentences: selection, strongest evidence, and key trade-off.]

## Evidence

*Rank-1 values copied exactly from <PRIMARY_RESULTS>; no new averages.*

[One two-row Markdown table: selected method and runner-up. Include
Method, Estimator, Rank, Fold_Num, Val_ROC_AUC, Test_ROC_AUC,
ROC_Gap, Val_F1, Test_F1, F1_Gap, Test_Precision,
Test_Recall/Sensitivity, and Test_Specificity. Use these exact
column names and source values. Mention Val_Precision,
Val_Recall/Sensitivity, or Val_Specificity only if they materially
change the comparison. Omit fields absent from the supplied rows.]



## Interpretation

[Two to four sentences comparing the methods using exact Rank-1
values and meaningful trade-offs. Discuss other folds only if they
materially affect the choice; do not call a Rank-1 value a fold
average. Make the explanation consistent with the Answer.]

{feature_explanation_rule}

## Limitations

- [One or two limitations supported by the supplied evidence.
   The test patients were held out from training and fold ranking,
   but using test scores to choose among methods means the selected
   test score is not an independent final evaluation of that choice.
   Do not claim external validation, clinical utility, or causation.]

## Provenance

- Unit: {unit_label}
- Decision model: [model name if supplied by the application;
  otherwise "not recorded in the input"]
- Selection source: <PRIMARY_RESULTS>; each method's Rank 1 was
  chosen by Val_ROC_AUC within the training pool
- Selected row: [exact Method, Estimator, Rank, and Fold_Num]
- Runner-up row: [exact Method, Estimator, Rank, and Fold_Num]
- Supporting source: <SUPPORTING_RESULTS> [name only the folds used]


Check every stated number against its source row. Do not invent
figures, paths, timestamps, costs, audits, or missing metrics.
"""
    return prompt


# The team's main.py appends this line to every decision prompt so
# the Provenance section names the LLM that made the decision.


def get_decision_model_note(model_name):
    return f"\n\nIn ## Provenance, write: - Decision model: {model_name}"
