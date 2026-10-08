"""Plan table parsing, versioning, discovery, and persistence."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


PLAN_PREFIX = "assembly_plan_"


def parse_plan_version(stem: str) -> tuple[str, int]:
    suffix = stem.replace(PLAN_PREFIX, "", 1)
    numbered = re.fullmatch(r"(.+)_regeneration_(\d+)", suffix)
    if numbered:
        return numbered.group(1), int(numbered.group(2))

    base = suffix
    legacy_count = 0
    while base.endswith("_regeneration"):
        base = base[: -len("_regeneration")]
        legacy_count += 1
    return base, legacy_count + 1


def plan_version_sort_key(path: Path) -> tuple[str, int, str]:
    base, generation = parse_plan_version(path.stem)
    return base, generation, path.name


def versioned_artifact_sort_key(filename: str, prefix: str, suffix: str):
    version_suffix = filename[len(prefix) : -len(suffix)]
    base, generation = parse_plan_version(f"{PLAN_PREFIX}{version_suffix}")
    return base, generation, filename


def next_regenerated_plan_path(input_csv_path: str | Path) -> Path:
    input_path = Path(input_csv_path)
    base, generation = parse_plan_version(input_path.stem)
    return input_path.with_name(f"{PLAN_PREFIX}{base}_regeneration_{generation + 1}.csv")


def find_latest_plan_csv(plans_dir: str | Path) -> tuple[Path, str]:
    plans_path = Path(plans_dir)
    csv_files = [p for p in plans_path.glob(f"{PLAN_PREFIX}*.csv") if p.is_file()]
    if not csv_files:
        raise FileNotFoundError("No assembly_plan_*.csv found in ./plans.")
    latest_csv = sorted(csv_files, key=plan_version_sort_key)[-1]
    timestamp = latest_csv.stem.replace(PLAN_PREFIX, "")
    return latest_csv, timestamp


def extract_phase4_table(text: str):
    print("\n==== [DEBUG] 开始提取Phase 4区块 ====")
    phase4_pattern = (
        r"(?:\s*(?:#+|\*\*)\s*)?Phase\s*4[^\n]*\n"
        r"([\s\S]+?)"
        r"(?=(?:\s*(?:#+|\*\*)\s*)?Phase\s*5|\Z)"
    )
    match = re.search(phase4_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        print("[DEBUG] Phase 4 正则未命中！")
        return None

    phase4_block = match.group(1).strip()
    table_pattern = re.compile(
        r"((?:\|[^\n]*?\|[^\n]*(?:\n|$))+?)"
        r"((?:\|\s*[-:]+\s*){2,}\|[^\n]*(?:\n|$))"
        r"((?:(?:\|[^\n]*?\|[^\n]*(?:\n|$))+)+)",
        re.MULTILINE,
    )
    table_match = table_pattern.search(phase4_block)
    if table_match:
        return (table_match.group(1) + table_match.group(2) + table_match.group(3)).strip()

    greedy_lines = [line for line in phase4_block.split("\n") if re.match(r"^\|.*\|\s*", line)]
    return "\n".join(greedy_lines) if len(greedy_lines) >= 2 else None


def extract_first_markdown_table(text: str):
    if not text or not isinstance(text, str):
        return None

    table_pattern = re.compile(
        r"((?:\|[^\n]*?\|[^\n]*(?:\n|$))+?)"
        r"((?:\|\s*[-:]+\s*){2,}\|[^\n]*(?:\n|$))"
        r"((?:(?:\|[^\n]*?\|[^\n]*(?:\n|$))+)+)",
        re.MULTILINE,
    )
    table_match = table_pattern.search(text)
    if table_match:
        return (table_match.group(1) + table_match.group(2) + table_match.group(3)).strip()

    block: list[str] = []
    for line in text.splitlines() + [""]:
        if re.match(r"^\s*\|.*\|\s*$", line):
            block.append(line.strip())
        elif len(block) >= 2:
            return "\n".join(block)
        else:
            block = []
    return None


def clean_markdown_table(markdown_table_text: str) -> list[list[str]]:
    lines = [
        line.strip()
        for line in markdown_table_text.strip().split("\n")
        if "|" in line
        and not re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", line)
        and not re.match(r"^\s*(Total|合计)", line, re.IGNORECASE)
    ]
    if len(lines) < 2:
        return []

    column_count = max(len([cell for cell in row.split("|") if cell.strip()]) for row in lines)
    table = []
    for row in lines:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        cells.extend([""] * (column_count - len(cells)))
        table.append(cells)
    return table


def write_csv(rows: Iterable[Iterable[str]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(rows)
    return path


def save_table_from_response(response: str, output_path: str | Path) -> Path:
    markdown_table = extract_phase4_table(response) or extract_first_markdown_table(response)
    if not markdown_table:
        raise ValueError("No valid Markdown table detected in model output.")
    rows = clean_markdown_table(markdown_table)
    if not rows:
        raise ValueError("Markdown table cleaning failed.")
    return write_csv(rows, output_path)

