
FEATURE_NAMES = {
    "baseline_diagnosis": "Baseline diagnosis",
    "CDRSB": "Clinical Dementia Rating Sum of Boxes",
    "CDGLOBAL": "Clinical Dementia Rating global score",
    "CDMEMORY": "Clinical Dementia Rating memory score",
    "TOTAL13": "ADAS-Cog 13 total score",
    "MMSCORE": "Mini-Mental State Examination score",
    "LDELTOTAL": "Logical Memory delayed recall score",
    "LIMMTOTAL": "Logical Memory immediate recall score",
    "CATANIMSC": "Category Fluency animal naming score",
    "TRAASCOR": "Trail Making Test Part A time",
    "TRABSCOR": "Trail Making Test Part B time",
    "FAQTOTAL": "Functional Activities Questionnaire score",
    "NPI": "Neuropsychiatric Inventory score",
    "GDTOTAL": "Geriatric Depression Scale score",
    "apoe_e4_count": "APOE e4 allele variant count",
    "gray_tcv": "Gray matter volume relative to total cranial volume",
    "white_tcv": "White matter volume relative to total cranial volume",
    "wmh_tcv": "White matter hyperintensity volume relative to total cranial volume",
    "PTGENDER": "Gender",
    "PTEDUCAT": "Years of education",
    "age_at_baseline": "Age at baseline",
}

# What each feature measures, written without parentheses so the
# LLM does not copy that style into the report.
FEATURE_DESCRIPTIONS = {
    # Diagnosis
    "baseline_diagnosis": (
        "Diagnosis at Year 0 (Y0), either cognitively normal (mark as 1) "
        "or mild cognitive impairment (mark as 2)."
    ),
    # Clinical Dementia Rating
    "CDRSB": "Global dementia staging scale from 0 to 18; higher means more impaired.",
    "CDGLOBAL": "Overall dementia stage from 0 to 3; higher means more impaired.",
    "CDMEMORY": "Memory portion of the Clinical Dementia Rating, 0 to 3; higher means more impaired.",
    # Cognitive tests
    "TOTAL13": "Broad cognitive test from 0 to 85; higher means more impaired.",
    "MMSCORE": "Brief cognitive screening test from 0 to 30; lower means more impaired.",
    "LDELTOTAL": "Recall of a short story after a delay; higher means better memory.",
    "LIMMTOTAL": "Immediate recall of a short story; higher means better memory.",
    "CATANIMSC": "A language test that records number of animals named in one minute, higher is better.",
    "TRAASCOR": "Seconds to connect numbered dots, a processing speed test; higher is slower.",
    "TRABSCOR": "Seconds to alternate numbers and letters, an executive function test; higher is slower.",
    # Function, behaviour, mood
    "FAQTOTAL": "Help needed with daily activities, 0 to 30; higher means more dependent.",
    "NPI": "Behavioural and psychiatric symptoms; higher means more symptoms.",
    "GDTOTAL": "Depressive symptoms, 0 to 15; higher means more symptoms.",
    # Genetics
    "apoe_e4_count": "An Alzheimer's risk factor gene variant that can either be 0, 1 or 2.",
    # MRI volumes
    "gray_tcv": "Brain gray matter volume scaled to head size.",
    "white_tcv": "Brain white matter volume scaled to head size.",
    "wmh_tcv": "Volume of white matter lesions on MRI, scaled to head size.",
    # Demographics
    "PTGENDER": "Gender - 1 denotes male, 2 denotes female.",
    "PTEDUCAT": "Years of formal education.",
    "age_at_baseline": "Age in years at baseline.",
}


def feature_name(feature):
    return FEATURE_NAMES.get(feature, feature)


def describe_features(features):
    return {
        feature: FEATURE_DESCRIPTIONS.get(feature, "no description")
        for feature in features
    }


# {"Clinical Dementia Rating Sum of Boxes": "Global dementia staging ...", ...}


def describe_features_by_name(features):
    return {
        feature_name(feature): description
        for feature, description in describe_features(features).items()
    }
