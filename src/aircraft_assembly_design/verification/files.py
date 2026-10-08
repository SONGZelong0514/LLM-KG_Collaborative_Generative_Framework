"""Runtime artifact discovery and serialization helpers."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import PATHS


PLAN_PREFIX = "assembly_plan_"
MBSE_PREFIX = "assembly_plan_MBSE_"
CONSTRAINT_PREFIX = "assembly_plan_design_constraint_"
REPORT_PREFIX = "assembly_plan_verification_report_"
SIM_RESULT_PREFIX = "assembly_plan_simulation_result_"


@dataclass
class LocatedFiles:
    timestamp: str
    plan_path: Path
    constraint_path: Optional[Path]
    matlab_path: Optional[Path]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def extract_timestamp(filename: str, prefix: str, suffix: str) -> Optional[str]:
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix) : -len(suffix)]
    return None


def parse_version_suffix(version_suffix: str) -> tuple[str, int]:
    numbered = re.fullmatch(r"(.+)_regeneration_(\d+)", version_suffix)
    if numbered:
        return numbered.group(1), int(numbered.group(2))
    base = version_suffix
    legacy_count = 0
    while base.endswith("_regeneration"):
        base = base[: -len("_regeneration")]
        legacy_count += 1
    return base, legacy_count + 1


def versioned_file_sort_key(path: Path, prefix: str, suffix: str):
    version_suffix = extract_timestamp(path.name, prefix, suffix) or path.stem
    base, generation = parse_version_suffix(version_suffix)
    return base, generation, path.name


def latest_file(directory: str | Path, prefix: str, suffix: str) -> Optional[Path]:
    directory = Path(directory)
    if not directory.exists():
        return None
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)
    ]
    return sorted(files, key=lambda path: versioned_file_sort_key(path, prefix, suffix))[-1] if files else None


def matching_or_latest(directory: str | Path, prefix: str, timestamp: str, suffix: str) -> Optional[Path]:
    directory = Path(directory)
    exact = directory / f"{prefix}{timestamp}{suffix}"
    if exact.exists():
        return exact
    base_timestamp, _ = parse_version_suffix(timestamp)
    base_exact = directory / f"{prefix}{base_timestamp}{suffix}"
    return base_exact if base_exact.exists() else latest_file(directory, prefix, suffix)


def locate_latest_inputs(
    plans_dir: str | Path = PATHS.plans,
    constraints_dir: str | Path = PATHS.constraints,
    simulation_dir: str | Path = PATHS.simulation,
) -> LocatedFiles:
    plan_path = latest_file(plans_dir, PLAN_PREFIX, ".csv")
    if plan_path is None:
        raise FileNotFoundError(
            "No assembly_plan_*.csv file found in ./plans. Please generate the plan first."
        )
    timestamp = extract_timestamp(plan_path.name, PLAN_PREFIX, ".csv")
    if not timestamp:
        raise ValueError(f"Cannot extract timestamp from plan file: {plan_path.name}")

    constraint_path = matching_or_latest(
        constraints_dir, CONSTRAINT_PREFIX, timestamp, ".txt"
    )
    matlab_path = Path(simulation_dir) / f"{MBSE_PREFIX}{timestamp}.m"
    if not matlab_path.exists():
        raise FileNotFoundError(
            f"Matching simulation model not found: {matlab_path}. "
            "Please run MBSE conversion and simulation conversion for the latest plan first."
        )
    return LocatedFiles(timestamp, plan_path, constraint_path, matlab_path)


def read_text_file(path: Optional[Path]) -> str:
    return "" if path is None or not path.exists() else path.read_text(encoding="utf-8", errors="ignore")


def read_plan_csv(plan_path: Path) -> list[dict[str, str]]:
    with plan_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(handle)
            if row
        ]


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty LLM response")
    clean = re.sub(r"```(?:json)?", "", text.strip(), flags=re.IGNORECASE).replace("```", "").strip()
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", clean):
        try:
            value, _ = decoder.raw_decode(clean[match.start() :])
            if isinstance(value, dict):
                candidates.append(value)
        except json.JSONDecodeError:
            continue

    for value in reversed(candidates):
        if (
            isinstance(value.get("violation_list"), list)
            and isinstance(value.get("repair_advice"), list)
        ):
            return value
    for value in reversed(candidates):
        if isinstance(value.get("violation_list"), list):
            value.setdefault("repair_advice", [])
            return value
    raise ValueError("No valid verification report JSON found in LLM output.")


def display_path(path: Optional[Path]) -> str:
    if path is None:
        return "None"
    try:
        return f"./{path.resolve().relative_to(PATHS.root.resolve()).as_posix()}"
    except ValueError:
        return str(path)

