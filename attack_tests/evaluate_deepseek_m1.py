import argparse
import csv
import math
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attack_tests.evaluate_static_vs_dynamic import TASK_TOPIC, SYSTEM_PROMPT
from llmagentprotector import PolymorphicPromptAssembler
from utils.attack_utils import get_payload
from utils.llm_utils import get_classifier

DEFAULT_MODEL_NAME = "deepseek-v4-pro"
MODEL_CHOICES = ("deepseek-v4-pro", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_NUM_ATTACKS = 100
DEFAULT_SEED = 1337
DEFAULT_SLEEP_SECONDS = 0.5
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "deepseek_m1_eval.csv"
CSV_COLUMNS = ("mode", "trial", "response_snippet", "classifier_label")
EVAL_MODES = ("static", "dynamic")
PAYLOAD_NAME = "M1_injection"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate M1_injection against DeepSeek with static and dynamic PPA separators."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL_NAME,
        help="DeepSeek model to evaluate.",
    )
    parser.add_argument(
        "--num-attacks",
        type=int,
        default=DEFAULT_NUM_ATTACKS,
        help="Number of requests to run per separator mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for reproducible static separator selection.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path.",
    )
    return parser


def _require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required to run this evaluation.")
    return api_key


def _build_client() -> OpenAI:
    return OpenAI(api_key=_require_api_key(), base_url=DEEPSEEK_BASE_URL)


def _response_snippet(response_text: str, limit: int = 160) -> str:
    normalized = re.sub(r"\s+", " ", (response_text or "")).strip()
    return normalized[:limit]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    phat = successes / total
    denominator = 1 + (z**2 / total)
    center = (phat + (z**2 / (2 * total))) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / total) + (z**2 / (4 * total**2)))
        / denominator
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def _call_deepseek(client: OpenAI, system_prompt: str, user_prompt: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
    )
    message = response.choices[0].message.content
    return message or ""


def evaluate_mode(
    *,
    client: OpenAI,
    mode: str,
    num_attacks: int,
    payload: str,
    classifier,
    seed: int,
    sleep_seconds: float,
    model_name: str,
) -> list[dict[str, str | int]]:
    protector = PolymorphicPromptAssembler(
        SYSTEM_PROMPT,
        TASK_TOPIC,
        separator_mode=mode,
        rng=random.Random(seed),
    )
    rows = []

    for trial in range(1, num_attacks + 1):
        session_id = f"deepseek-m1-{mode}-{trial}"
        system_prompt, user_prompt, _ = protector.double_prompt_assemble(
            payload,
            session_id=session_id,
        )
        response = _call_deepseek(client, system_prompt, user_prompt, model_name)
        classifier_label = classifier(response)
        rows.append(
            {
                "mode": mode,
                "trial": trial,
                "response_snippet": _response_snippet(response),
                "classifier_label": classifier_label,
            }
        )
        if trial < num_attacks and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows


def summarize_rows(rows: list[dict[str, str | int]]) -> list[dict[str, float | int | str]]:
    summary_rows = []
    for mode in EVAL_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        attacked = sum(1 for row in mode_rows if row["classifier_label"] == "Attacked")
        total = len(mode_rows)
        ci_lower, ci_upper = wilson_interval(attacked, total)
        summary_rows.append(
            {
                "mode": mode,
                "N": total,
                "ASR": attacked / total if total else 0.0,
                "Wilson CI lower": ci_lower,
                "Wilson CI upper": ci_upper,
            }
        )
    return summary_rows


def write_results(rows: list[dict[str, str | int]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary(summary_rows: list[dict[str, float | int | str]]) -> None:
    header = (
        f"{'mode':<7} | {'N':>3} | {'ASR':>5} | {'Wilson CI lower':>15} | {'Wilson CI upper':>15}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['mode']:<7} | {row['N']:>3} | {row['ASR']:>5.3f} | "
            f"{row['Wilson CI lower']:>15.3f} | {row['Wilson CI upper']:>15.3f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.num_attacks < 0:
        raise ValueError("--num-attacks must be non-negative")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative")

    payload = get_payload(PAYLOAD_NAME)
    classifier = get_classifier(PAYLOAD_NAME)
    client = _build_client()

    all_rows = []
    for mode in EVAL_MODES:
        all_rows.extend(
            evaluate_mode(
                client=client,
                mode=mode,
                num_attacks=args.num_attacks,
                payload=payload,
                classifier=classifier,
                seed=args.seed,
                sleep_seconds=args.sleep_seconds,
                model_name=args.model,
            )
        )

    output_path = write_results(all_rows, args.output)
    print_summary(summarize_rows(all_rows))
    print(f"Saved per-trial results to {output_path}")


if __name__ == "__main__":
    main()