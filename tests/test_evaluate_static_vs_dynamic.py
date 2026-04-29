import argparse

from attack_tests.evaluate_static_vs_dynamic import (
    apply_env_defaults,
    build_phase5_interpretation,
    parse_modes,
    summarize_results,
    summarize_leakage_blast_radius,
)
from utils.attack_utils import get_payload, list_payloads


def test_payload_registry_contains_expected_attack_families():
    payloads = list_payloads()

    assert "baseline_salad" in payloads
    assert "direct_override_salad" in payloads
    assert "role_confusion_salad" in payloads
    assert "format_breakout_salad" in payloads
    assert "separator_echo_salad" in payloads
    assert "benign_summary" in payloads
    assert "salad" in get_payload("baseline_salad").lower()


def test_parse_modes_rejects_unknown_modes():
    assert parse_modes("static,dynamic") == ["static", "dynamic"]

    try:
        parse_modes("static,unknown")
    except ValueError as exc:
        assert "Unknown mode" in str(exc)
    else:
        raise AssertionError("parse_modes should reject unknown modes")


def test_summarize_results_splits_target_and_classifier_failures():
    details = [
        {"classification": "Attacked", "leak_detected": True, "provider_error": None},
        {"classification": "Defended", "leak_detected": False, "provider_error": None},
        {
            "classification": "ProviderBlocked",
            "leak_detected": False,
            "provider_error": "target_model:BadRequestError: blocked",
        },
        {
            "classification": "Unknown",
            "leak_detected": False,
            "provider_error": "classifier:RuntimeError: unavailable",
        },
    ]

    summary = summarize_results("static", 4, "llama", details)

    assert summary["evaluated_attempts"] == 3
    assert summary["attacked"] == 1
    assert summary["defended"] == 1
    assert summary["unknown"] == 1
    assert summary["provider_blocked"] == 1
    assert summary["target_provider_blocked"] == 1
    assert summary["classifier_failed"] == 1
    assert summary["provider_error_breakdown"] == {"target_model": 1, "classifier": 1}
    assert summary["asr"] == 1 / 3
    assert summary["defense_rate"] == 1 / 3
    assert summary["leak_rate"] == 1 / 3


def test_summarize_results_handles_zero_evaluated_attempts():
    details = [
        {
            "classification": "ProviderBlocked",
            "leak_detected": False,
            "provider_error": "target_model:BadRequestError: blocked",
        }
    ]

    summary = summarize_results("dynamic", 1, "llama", details)

    assert summary["evaluated_attempts"] == 0
    assert summary["asr"] == 0
    assert summary["defense_rate"] == 0
    assert summary["leak_rate"] == 0
    assert summary["provider_blocked_rate"] == 1


def test_apply_env_defaults_reports_defaults(monkeypatch):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        num_attacks=None,
        model_backend=None,
        modes=None,
        payload=None,
        run_id=None,
        seed=None,
    )
    monkeypatch.delenv("PPA_NUM_ATTACKS", raising=False)
    monkeypatch.delenv("PPA_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("PPA_EVAL_MODES", raising=False)
    monkeypatch.delenv("PPA_PAYLOAD", raising=False)
    monkeypatch.delenv("PPA_RUN_ID", raising=False)
    monkeypatch.delenv("PPA_RANDOM_SEED", raising=False)

    resolved = apply_env_defaults(args, parser)

    assert resolved.num_attacks == 100
    assert resolved.model_backend == "gpt"
    assert resolved.modes == ["static", "dynamic"]
    assert resolved.payload == "baseline_salad"
    assert resolved.seed == 1337


def test_summarize_leakage_blast_radius_marks_static_pool_reuse():
    details = [
        {
            "session_id": "static-1",
            "canary": {"left": "LEFT", "right": "RIGHT"},
            "leak_detected": True,
            "leak_detail": {"left": True, "right": False, "detected": True},
        },
        {
            "session_id": "static-2",
            "canary": {"left": "LEFT", "right": "RIGHT"},
            "leak_detected": False,
            "leak_detail": {"left": False, "right": False, "detected": False},
        },
    ]

    summary = summarize_leakage_blast_radius("static", details)

    assert summary["blast_radius"] == "pool-reusable"
    assert summary["leaked_sessions"] == ["static-1"]
    assert summary["unique_leaked_separator_count"] == 1
    assert summary["reused_separator_count"] == 2
    assert summary["max_observed_separator_reuse"] == 2


def test_build_phase5_interpretation_separates_asr_parity_from_leakage_scope():
    results = {
        "results": {
            "static": {
                "summary": {"asr": 0.0, "leak_rate": 0.05},
                "details": [
                    {
                        "session_id": "static-1",
                        "canary": ["STATIC", "STATIC"],
                        "leak_detected": True,
                        "leak_detail": {"left": True, "right": True, "detected": True},
                    }
                ],
            },
            "dynamic": {
                "summary": {"asr": 0.0, "leak_rate": 0.0},
                "details": [
                    {
                        "session_id": "dynamic-1",
                        "canary": {"left": "DYN-L", "right": "DYN-R"},
                        "leak_detected": False,
                        "leak_detail": {"left": False, "right": False, "detected": False},
                    }
                ],
            },
        }
    }

    interpretation = build_phase5_interpretation(results)

    assert interpretation["comparison"]["asr_delta_dynamic_minus_static"] == 0.0
    assert interpretation["comparison"]["attack_resistance_parity"] is True
    assert interpretation["comparison"]["leak_rate_delta_dynamic_minus_static"] == -0.05
    assert interpretation["leakage_blast_radius"]["static"]["blast_radius"] == "pool-reusable"
    assert interpretation["leakage_blast_radius"]["dynamic"]["blast_radius"] == "request-scoped"