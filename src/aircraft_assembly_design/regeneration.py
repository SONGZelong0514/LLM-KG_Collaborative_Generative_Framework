"""Feedback-driven assembly-plan regeneration."""

from __future__ import annotations

import json
from pathlib import Path

from .clients import get_openai_client
from .config import PATHS, SETTINGS
from .plans import find_latest_plan_csv, next_regenerated_plan_path, save_table_from_response
from .prompts import build_regeneration_prompt
from .streaming import extract_stream_token


def read_plan_table(csv_path: str | Path) -> str:
    return Path(csv_path).read_text(encoding="utf-8-sig")


def read_verification_report(timestamp: str, verification_dir: str | Path = PATHS.verification):
    report_path = Path(verification_dir) / f"assembly_plan_verification_report_{timestamp}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Verification report not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8")), report_path


def regeneration_token_stream(human_feedback: str = ""):
    latest_csv, timestamp = find_latest_plan_csv(PATHS.plans)
    plan_table = read_plan_table(latest_csv)
    verification_report, report_path = read_verification_report(timestamp)
    prompt_text = build_regeneration_prompt(plan_table, verification_report, human_feedback)

    print("\n========== Regeneration LLM Context ==========")
    print(prompt_text)
    print("========== End Regeneration LLM Context ==========\n")

    stream = get_openai_client().chat.completions.create(
        model=SETTINGS.model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=SETTINGS.temperature,
        stream=True,
    )
    full_output = ""
    for chunk in stream:
        token = extract_stream_token(chunk, hide_thinking=True)
        if token:
            full_output += token
            print(token, end="", flush=True)
            yield token, full_output, timestamp, latest_csv, report_path


def save_regenerated_csv(full_output: str, input_csv_path: str | Path) -> Path:
    output_path = next_regenerated_plan_path(input_csv_path)
    return save_table_from_response(full_output, output_path)


def regeneration_action(human_feedback, history):
    history = history or []
    human_feedback = human_feedback or ""
    history = history + [
        {"role": "user", "content": f"Human feedback:\n{human_feedback}"},
        {"role": "assistant", "content": "🔁 Running regeneration...\n\n"},
    ]
    yield history

    full_output = ""
    latest_csv = None
    report_path = None
    try:
        for token, full_output, _, latest_csv, report_path in regeneration_token_stream(human_feedback):
            history[-1]["content"] += token
            yield history

        output_path = save_regenerated_csv(full_output, latest_csv)
        message = (
            "\n\n✅ **Regeneration completed.**\n\n"
            f"Input plan: `./plans/{latest_csv.name}`\n"
            f"Input verification report: `./Verification/{report_path.name}`\n"
            f"Regenerated plan saved as: `./plans/{output_path.name}`"
        )
        history[-1]["content"] += message
        yield history
    except Exception as exc:
        history[-1]["content"] += f"\n\n❌ Regeneration failed: {exc}"
        yield history

