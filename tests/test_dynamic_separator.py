import re

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

    assert re.fullmatch(r"====[0-9a-f]{16}====", separator)
    assert "tenant-a-user-42" not in separator
    assert "\n" not in separator


def test_generate_separator_pair_uses_current_canary_shape():
    left_sep, right_sep = generate_separator_pair(session_id="session-1")

    assert left_sep == right_sep
    assert left_sep.startswith("====")
    assert left_sep.endswith("====")


def test_static_mode_still_selects_from_config_pool(monkeypatch):
    static_pair = ["LEFT_STATIC", "RIGHT_STATIC"]
    monkeypatch.setattr(config, "SEPARATORS", [static_pair])

    protector = PolymorphicPromptAssembler(SYSTEM_PROMPT, TASK_TOPIC)
    _, user_prompt, canary = protector.double_prompt_assemble(USER_INPUT)

    assert canary == tuple(static_pair)
    assert "LEFT_STATIC" in user_prompt
    assert "RIGHT_STATIC" in user_prompt


def test_dynamic_mode_embeds_generated_separator_in_single_prompt():
    dynamic_pair = ("====dynamic-test====", "====dynamic-test====")
    protector = PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_provider=lambda session_id=None: dynamic_pair,
    )

    secure_prompt, canary = protector.single_prompt_assemble(USER_INPUT, session_id="session-1")

    assert canary == dynamic_pair
    assert secure_prompt.count("====dynamic-test====") >= 2
    assert USER_INPUT in secure_prompt


def test_dynamic_canary_leak_detect():
    dynamic_pair = ("====dynamic-test====", "====dynamic-test====")
    protector = PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_provider=lambda session_id=None: dynamic_pair,
    )

    assert protector.leak_detect("response leaked ====dynamic-test====", dynamic_pair) is True
    assert protector.leak_detect("ordinary response", dynamic_pair) is False
