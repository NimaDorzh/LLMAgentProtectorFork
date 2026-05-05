import argparse

from attack_tests.evaluate_static_vs_dynamic import (
    apply_env_defaults,
    build_sweep_summary,
    build_phase5_interpretation,
    default_output_path,
    parse_modes,
    parse_payloads,
    summarize_results,
    summarize_leakage_blast_radius,
)
from utils.attack_utils import get_payload, list_payloads


def test_payload_registry_contains_expected_attack_families():
    payloads = list_payloads()

    assert "baseline_salad" in payloads
    assert "M1_injection" in payloads
    assert "M2_injection" in payloads
    assert "direct_override_salad" in payloads
    assert "role_confusion_salad" in payloads
    assert "format_breakout_salad" in payloads
    assert "separator_echo_salad" in payloads
    assert "benign_summary" in payloads
    assert "salad" in get_payload("baseline_salad").lower()
    assert "injection successful" in get_payload("M1_injection").lower()
    assert "injection successful" in get_payload("M2_injection").lower()


def test_parse_modes_rejects_unknown_modes():
    assert parse_modes("static,dynamic") == ["static", "dynamic"]

    try:
        parse_modes("static,unknown")
    except ValueError as exc:
        assert "Unknown mode" in str(exc)
    else:
        raise AssertionError("parse_modes should reject unknown modes")


def test_parse_payloads_supports_defaults_all_and_rejects_unknown_payloads():
    assert parse_payloads("") == ["baseline_salad"]
    assert parse_payloads("baseline_salad,M1_injection") == ["baseline_salad", "M1_injection"]
    assert parse_payloads("all") == list_payloads()

    try:
        parse_payloads("baseline_salad,unknown")
    except ValueError as exc:
        assert "Unknown payload" in str(exc)
    else:
        raise AssertionError("parse_payloads should reject unknown payload names")


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
        payloads=None,
        payload=None,
        run_id=None,
        seed=None,
    )
    monkeypatch.delenv("PPA_NUM_ATTACKS", raising=False)
    monkeypatch.delenv("PPA_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("PPA_EVAL_MODES", raising=False)
    monkeypatch.delenv("PPA_PAYLOADS", raising=False)
    monkeypatch.delenv("PPA_PAYLOAD", raising=False)
    monkeypatch.delenv("PPA_RUN_ID", raising=False)
    monkeypatch.delenv("PPA_RANDOM_SEED", raising=False)

    resolved = apply_env_defaults(args, parser)

    assert resolved.num_attacks == 100
    assert resolved.model_backend == "gpt"
    assert resolved.modes == ["static", "dynamic"]
    assert resolved.payloads == ["baseline_salad"]
    assert resolved.payload == "baseline_salad"
    assert resolved.seed == 1337


def test_apply_env_defaults_uses_payloads_env_and_converts_legacy_payload(monkeypatch):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        num_attacks=5,
        model_backend="llama",
        modes=["static"],
        payloads=None,
        payload=None,
        run_id="demo",
        seed=7,
    )
    monkeypatch.setenv("PPA_PAYLOADS", "M1_injection,M2_injection")
    monkeypatch.delenv("PPA_PAYLOAD", raising=False)

    resolved = apply_env_defaults(args, parser)

    assert resolved.payloads == ["M1_injection", "M2_injection"]
    assert resolved.payload is None

    legacy_args = argparse.Namespace(
        num_attacks=5,
        model_backend="llama",
        modes=["static"],
        payloads=None,
        payload="M1_injection",
        run_id="demo",
        seed=7,
    )

    resolved_legacy = apply_env_defaults(legacy_args, parser)

    assert resolved_legacy.payloads == ["M1_injection"]
    assert resolved_legacy.payload == "M1_injection"


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
    mode_results = {
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

    interpretation = build_phase5_interpretation(mode_results)

    assert interpretation["comparison"]["asr_delta_dynamic_minus_static"] == 0.0
    assert interpretation["comparison"]["attack_resistance_parity"] is True
    assert interpretation["comparison"]["leak_rate_delta_dynamic_minus_static"] == -0.05
    assert interpretation["leakage_blast_radius"]["static"]["blast_radius"] == "pool-reusable"
    assert interpretation["leakage_blast_radius"]["dynamic"]["blast_radius"] == "request-scoped"


def test_build_sweep_summary_flattens_payload_and_mode_metrics():
    all_payload_results = {
        "baseline_salad": {
            "static": {
                "summary": {
                    "total_attempts": 10,
                    "evaluated_attempts": 9,
                    "attacked": 1,
                    "defended": 8,
                    "unknown": 0,
                    "asr": 1 / 9,
                    "defense_rate": 8 / 9,
                    "leak_count": 0,
                    "leak_rate": 0.0,
                    "provider_blocked": 1,
                    "provider_blocked_rate": 0.1,
                    "target_provider_blocked": 1,
                    "classifier_failed": 0,
                }
            },
            "phase5_interpretation": {},
        }
    }

    sweep_summary = build_sweep_summary(all_payload_results)

    assert sweep_summary == [
        {
            "payload_name": "baseline_salad",
            "mode": "static",
            "total_attempts": 10,
            "evaluated_attempts": 9,
            "attacked": 1,
            "defended": 8,
            "unknown": 0,
            "asr": 1 / 9,
            "defense_rate": 8 / 9,
            "leak_count": 0,
            "leak_rate": 0.0,
            "provider_blocked": 1,
            "provider_blocked_rate": 0.1,
            "target_provider_blocked": 1,
            "classifier_failed": 0,
        }
    ]


def test_default_output_path_switches_for_multi_payload_runs():
    assert default_output_path("llama", "run42", ["baseline_salad"]).name == "static_vs_dynamic_llama_run42.json"
    assert (
        default_output_path("llama", "run42", ["baseline_salad", "M1_injection"]).name
        == "multi_payload_sweep_llama_run42.json"
    )