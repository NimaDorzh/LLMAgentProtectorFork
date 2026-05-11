from attack_tests.evaluate_local_classifiers import (
    BENIGN_CONTROLS,
    compute_summary_table,
    extract_top_label,
    infer_injection_label,
    label_looks_injection,
)


def test_label_looks_injection_handles_common_label_names():
    assert label_looks_injection("PROMPT_INJECTION") is True
    assert label_looks_injection("jailbreak") is True
    assert label_looks_injection("BENIGN") is False
    assert label_looks_injection("safe") is False


def test_extract_top_label_supports_nested_pipeline_output():
    prediction = [[{"label": "BENIGN", "score": 0.25}, {"label": "INJECTION", "score": 0.75}]]

    assert extract_top_label(prediction) == "INJECTION"


def test_infer_injection_label_uses_probe_outputs():
    def fake_classifier(text, **kwargs):
        if "INJECTION SUCCESS" in text:
            return [
                {"label": "PROMPT_INJECTION", "score": 0.93},
                {"label": "BENIGN", "score": 0.07},
            ]
        return [
            {"label": "BENIGN", "score": 0.89},
            {"label": "PROMPT_INJECTION", "score": 0.11},
        ]

    assert infer_injection_label(fake_classifier, "fake") == "PROMPT_INJECTION"


def test_summary_uses_attack_rows_and_benign_controls_for_fpr():
    payload_rows = [
        {"payload_name": "baseline_salad", "meta_label": "INJECTION", "protectai_label": "BENIGN", "deepset_label": "INJECTION"},
        {"payload_name": "naive_injection", "meta_label": "BENIGN", "protectai_label": "INJECTION", "deepset_label": "INJECTION"},
        {"payload_name": "benign_summary", "meta_label": "BENIGN", "protectai_label": "BENIGN", "deepset_label": "BENIGN"},
    ]
    benign_results = {
        "meta": ["BENIGN"] * 19 + ["INJECTION"],
        "protectai": ["BENIGN"] * 20,
        "deepset": ["BENIGN"] * 18 + ["INJECTION", "INJECTION"],
    }

    summary = compute_summary_table(payload_rows, benign_results)
    summary_by_name = {row["classifier"]: row for row in summary}

    assert len(BENIGN_CONTROLS) == 20
    assert summary_by_name["meta"]["ASR_not_caught (%)"] == 50.0
    assert summary_by_name["meta"]["FPR"] == 0.05
    assert summary_by_name["protectai"]["ASR_not_caught (%)"] == 50.0
    assert summary_by_name["protectai"]["FPR"] == 0.0
    assert summary_by_name["deepset"]["ASR_not_caught (%)"] == 0.0
    assert summary_by_name["deepset"]["FPR"] == 0.1