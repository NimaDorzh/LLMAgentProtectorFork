import re

import pytest

from llmagentprotector import PolymorphicPromptAssembler, generate_separator, generate_separator_pair
from llmagentprotector import config


SYSTEM_PROMPT = "Summarize the user input.\n{user_input}\n"
TASK_TOPIC = "summarize the user input"
USER_INPUT = "Half Moon Bay is scenic."


def test_generate_separator_is_hash_marker_without_raw_session_id():
    separator = generate_separator(
        session_id="tenant-a-user-42",
        timestamp_ns=123456789,
        nonce="fixed-nonce",
    )

    assert re.fullmatch(r"====[0-9a-f]{24}====", separator)
    assert "tenant-a-user-42" not in separator
    assert "\n" not in separator


def test_generate_separator_pair_uses_current_canary_shape():
    left_sep, right_sep = generate_separator_pair(
        session_id="session-1",
        timestamp_ns=123456789,
        nonce="fixed-nonce",
    )

    assert left_sep != right_sep
    assert re.fullmatch(r"====BEGIN-[0-9a-f]{24}====", left_sep)
    assert re.fullmatch(r"====END-[0-9a-f]{24}====", right_sep)
    assert "session-1" not in left_sep
    assert "session-1" not in right_sep


def test_generate_separator_pair_is_fresh_for_same_session_id():
    first_pair = generate_separator_pair(session_id="stable-session")
    second_pair = generate_separator_pair(session_id="stable-session")

    assert first_pair != second_pair


def test_prefix_length_boundaries_are_enforced():
    assert re.fullmatch(
        r"====[0-9a-f]{64}====",
        generate_separator(session_id="session", timestamp_ns=1, nonce="nonce", prefix_length=64),
    )
    left_sep, right_sep = generate_separator_pair(
        session_id="session",
        timestamp_ns=1,
        nonce="nonce",
        prefix_length=64,
    )
    assert re.fullmatch(r"====BEGIN-[0-9a-f]{64}====", left_sep)
    assert re.fullmatch(r"====END-[0-9a-f]{64}====", right_sep)

    for invalid_prefix_length in (0, 65):
        with pytest.raises(ValueError):
            generate_separator(prefix_length=invalid_prefix_length)
        with pytest.raises(ValueError):
            generate_separator_pair(prefix_length=invalid_prefix_length)


def test_static_mode_still_selects_from_config_pool(monkeypatch):
    static_pair = ["LEFT_STATIC", "RIGHT_STATIC"]
    monkeypatch.setattr(config, "SEPARATORS", [static_pair])

    protector = PolymorphicPromptAssembler(SYSTEM_PROMPT, TASK_TOPIC)
    _, user_prompt, canary = protector.double_prompt_assemble(USER_INPUT)

    assert canary == tuple(static_pair)
    assert "LEFT_STATIC" in user_prompt
    assert "RIGHT_STATIC" in user_prompt


def test_dynamic_mode_embeds_generated_separator_in_single_prompt():
    dynamic_pair = ("====dynamic-left====", "====dynamic-right====")
    protector = PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_provider=lambda session_id=None: dynamic_pair,
    )

    secure_prompt, canary = protector.single_prompt_assemble(USER_INPUT, session_id="session-1")

    assert canary == dynamic_pair
    assert "====dynamic-left====" in secure_prompt
    assert "====dynamic-right====" in secure_prompt
    assert USER_INPUT in secure_prompt


def test_dynamic_mode_embeds_generated_separator_in_double_prompt():
    protector = PolymorphicPromptAssembler(
        "Summarize the user input.",
        TASK_TOPIC,
        separator_mode="dynamic",
    )

    secure_system_prompt, secure_user_prompt, canary = protector.double_prompt_assemble(
        USER_INPUT,
        session_id="session-1",
    )

    left_sep, right_sep = canary
    assert left_sep != right_sep
    assert re.fullmatch(r"====BEGIN-[0-9a-f]{24}====", left_sep)
    assert re.fullmatch(r"====END-[0-9a-f]{24}====", right_sep)
    assert left_sep in secure_user_prompt
    assert right_sep in secure_user_prompt
    assert USER_INPUT in secure_user_prompt
    assert "session-1" not in secure_user_prompt
    assert secure_system_prompt.startswith("Summarize the user input.")


def test_double_prompt_assemble_does_not_store_request_prompt_on_instance():
    protector = PolymorphicPromptAssembler(
        "Summarize the user input.",
        TASK_TOPIC,
        separator_provider=lambda session_id=None: ("LEFT", "RIGHT"),
    )

    _, secure_user_prompt, _ = protector.double_prompt_assemble("first request")

    assert "first request" in secure_user_prompt
    assert not hasattr(protector, "secure_user_prompt")


def test_dynamic_canary_leak_detect():
    dynamic_pair = ("====dynamic-left====", "====dynamic-right====")
    protector = PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_provider=lambda session_id=None: dynamic_pair,
    )

    assert protector.leak_detect("response leaked ====dynamic-left====", dynamic_pair) is True
    assert protector.leak_detect("response leaked ====dynamic-right====", dynamic_pair) is True
    assert protector.leak_detect("ordinary response", dynamic_pair) is False


def test_leak_detect_detail_reports_leaked_side():
    dynamic_pair = ("====dynamic-left====", "====dynamic-right====")
    protector = PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_provider=lambda session_id=None: dynamic_pair,
    )

    assert protector.leak_detect_detail("leaked ====dynamic-left====", dynamic_pair) == {
        "left": True,
        "right": False,
        "detected": True,
    }
    assert protector.leak_detect_detail("leaked ====dynamic-right====", dynamic_pair) == {
        "left": False,
        "right": True,
        "detected": True,
    }
    assert protector.leak_detect_detail("ordinary response", dynamic_pair) == {
        "left": False,
        "right": False,
        "detected": False,
    }


def test_single_prompt_without_placeholder_raises_error():
    protector = PolymorphicPromptAssembler(
        "Summarize the user input.",
        TASK_TOPIC,
        separator_provider=lambda session_id=None: ("LEFT", "RIGHT"),
    )

    with pytest.raises(ValueError, match=r"\{user_input\}"):
        protector.single_prompt_assemble(USER_INPUT)
