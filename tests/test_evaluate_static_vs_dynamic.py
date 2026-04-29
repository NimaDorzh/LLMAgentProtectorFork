import argparse

from attack_tests.evaluate_static_vs_dynamic import apply_env_defaults, parse_modes, summarize_results
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