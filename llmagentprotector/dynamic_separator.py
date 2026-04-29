import hashlib
import secrets
import time
from uuid import uuid4


DEFAULT_SEPARATOR_PREFIX_LENGTH = 24
DEFAULT_PAIR_PREFIX_LENGTH = 24
MAX_SEPARATOR_LENGTH = 80


def validate_separator(separator: str) -> str:
    if not isinstance(separator, str):
        raise TypeError("separator must be a string")
    if not separator:
        raise ValueError("separator must be non-empty")
    if "\n" in separator or "\r" in separator:
        raise ValueError("separator must be single-line")
    if len(separator) > MAX_SEPARATOR_LENGTH:
        raise ValueError(f"separator must be at most {MAX_SEPARATOR_LENGTH} characters")
    return separator


def generate_separator(
    session_id: str | None = None,
    *,
    timestamp_ns: int | None = None,
    nonce: str | None = None,
    prefix_length: int = DEFAULT_SEPARATOR_PREFIX_LENGTH,
) -> str:
    if prefix_length <= 0 or prefix_length > 64:
        raise ValueError("prefix_length must be between 1 and 64")

    private_session_id = session_id or str(uuid4())
    private_timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
    private_nonce = nonce or secrets.token_hex(16)
    seed = f"{private_timestamp_ns}:{private_session_id}:{private_nonce}".encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()[:prefix_length]
    return validate_separator(f"===={digest}====")


def _generate_tagged_separator(
    tag: str,
    session_id: str,
    timestamp_ns: int,
    nonce: str,
    prefix_length: int,
) -> str:
    seed = f"{tag}:{timestamp_ns}:{session_id}:{nonce}".encode("utf-8")
    digest = hashlib.sha256(seed).hexdigest()[:prefix_length]
    return validate_separator(f"===={tag}-{digest}====")


def generate_separator_pair(
    session_id: str | None = None,
    *,
    timestamp_ns: int | None = None,
    nonce: str | None = None,
    prefix_length: int = DEFAULT_PAIR_PREFIX_LENGTH,
) -> tuple[str, str]:
    if prefix_length <= 0 or prefix_length > 64:
        raise ValueError("prefix_length must be between 1 and 64")

    private_session_id = session_id or str(uuid4())
    private_timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
    private_nonce = nonce or secrets.token_hex(16)
    return (
        _generate_tagged_separator("BEGIN", private_session_id, private_timestamp_ns, private_nonce, prefix_length),
        _generate_tagged_separator("END", private_session_id, private_timestamp_ns, private_nonce, prefix_length),
    )


class DynamicSeparatorProvider:
    """Generate a fresh per-call separator pair, even when the session id is stable."""

    def __init__(self, default_session_id: str | None = None):
        self.default_session_id = default_session_id

    def __call__(self, session_id: str | None = None) -> tuple[str, str]:
        return generate_separator_pair(session_id=session_id or self.default_session_id)
