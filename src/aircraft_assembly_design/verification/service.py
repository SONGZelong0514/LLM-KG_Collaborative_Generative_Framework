"""Orchestrate deterministic checks and LLM-based constraint verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..clients import get_openai_client
from ..config import PATHS, SETTINGS
from ..prompts import build_verification_prompt
from ..streaming import extract_stream_token
from .checks import (
    check_operation_costs,
    check_operation_durations,
    check_operation_resources,
    check_resource_peak_usage,
)
from .files import (
    REPORT_PREFIX,
    SIM_RESULT_PREFIX,
    display_path,
    ensure_dir,
    extract_json_object,
    extract_timestamp,
    locate_latest_inputs,
    read_plan_csv,
    read_text_file,
    PLAN_PREFIX,
)
from .knowledge import (
    query_operation_dependencies,
    query_operation_knowledge,
    query_operation_resource_requirements,
    query_resource_knowledge,
)


def build_simulation_result_json(
    plan_path: Path,
    matlab_path: Optional[Path] = None,
    output_dir: str | Path = PATHS.simulation,
) -> dict[str, Any]:
    del matlab_path  # The generated model is required by the workflow but checks use plan and KG data.
    plan_rows = read_plan_csv(plan_path)
    timestamp = extract_timestamp(plan_path.name, PLAN_PREFIX, ".csv") or "unknown"

    print("\n========== BUILD SIMULATION RESULT ==========")
    print(f"[DEBUG] Plan CSV: {plan_path}")
    resources = query_resource_knowledge()
    operations = query_operation_knowledge()
    requirements = query_operation_resource_requirements()

    result = {
        "operation_duration_check": check_operation_durations(plan_rows, operations),
        "operation_resource_check": check_operation_resources(plan_rows, requirements),
        "operation_cost_check": check_operation_costs(plan_rows, operations, resources, requirements),
        "resource_usage_check": check_resource_peak_usage(plan_rows, resources),
    }
    output_path = ensure_dir(output_dir) / f"{SIM_RESULT_PREFIX}{timestamp}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def verification_token_stream(prompt_text: str):
    stream = get_openai_client().chat.completions.create(
        model=SETTINGS.model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=SETTINGS.temperature,
        stream=True,
    )
    for chunk in stream:
        token = extract_stream_token(chunk, hide_thinking=True)
        if token:
            yield token


def run_verification_stream():
    print("\n================ VERIFICATION START ================")
    files = locate_latest_inputs()
    plan_path, constraint_path, matlab_path, timestamp = (
        files.plan_path,
        files.constraint_path,
        files.matlab_path,
        files.timestamp,
    )
    yield (
        f"📄 Plan file: {display_path(plan_path)}\n"
        f"📄 Constraint file: {display_path(constraint_path)}\n"
        f"📄 MATLAB file: {display_path(matlab_path)}\n\n"
    )

    simulation_result = build_simulation_result_json(plan_path, matlab_path)
    simulation_text = json.dumps(simulation_result, ensure_ascii=False, indent=2)
    yield (
        "📊 Simulation Result (for verification evidence):\n"
        "----------------------------------------\n"
        f"{simulation_text}\n"
        "----------------------------------------\n\n"
    )

    operation_dependency = json.dumps(
        query_operation_dependencies(), ensure_ascii=False, indent=2
    )
    prompt = build_verification_prompt(
        operation_dependency,
        read_text_file(constraint_path),
        read_text_file(plan_path),
    )
    yield "🤖 Evaluator_LLM is verifying the compliance of engineering constraints...\n\n"

    full_output = ""
    for token in verification_token_stream(prompt):
        full_output += token
        print(token, end="", flush=True)
        yield token

    try:
        report = extract_json_object(full_output)
    except Exception as exc:
        report = {
            "violation_list": [{
                "id": "PARSE_ERROR",
                "involved_operations": [],
                "reason": "Evaluator_LLM output could not be parsed as the required verification report JSON.",
                "evidence": str(exc),
            }],
            "repair_advice": [{
                "id": "R_PARSE_ERROR",
                "advice": "Regenerate the verification report strictly following the required JSON schema.",
            }],
        }

    report_path = ensure_dir(PATHS.verification) / f"{REPORT_PREFIX}{timestamp}.json"
    report_path.write_text(
        json.dumps(
            {"verification_report": report, "simulation_result": simulation_result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    yield f"\n\n✅ Verification saved: {display_path(report_path)}"


def run_verification() -> str:
    return "".join(str(chunk) for chunk in run_verification_stream())

