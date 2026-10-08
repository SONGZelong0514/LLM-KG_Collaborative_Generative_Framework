"""Pure deterministic checks used by plan verification."""

from __future__ import annotations

import re
from typing import Any


RESOURCE_ALIAS = {
    "Light Flex Track Rail": "Light Flex Track Rail",
    "Light Flexible Track Rail": "Light Flex Track Rail",
    "Light Flex Track Robot": "Light Flex Track Robot",
    "Light Flexible Track Robot": "Light Flex Track Robot",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip().replace("€", "").replace(",", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(numbers[0]) if numbers else default


def _canonical_resource_name(name: Any) -> str:
    if name is None:
        return ""
    text = " ".join(str(name).strip().split())
    return RESOURCE_ALIAS.get(text, text)


def _parse_required_resources(resource_text: str) -> dict[str, int]:
    resources: dict[str, int] = {}
    for name, quantity in re.findall(r"([^,;()]+?)\s*\((\d+)\)", resource_text or ""):
        canonical = _canonical_resource_name(" ".join(name.strip().strip(";").split()))
        resources[canonical] = resources.get(canonical, 0) + int(quantity)
    return resources


def _compute_resource_peaks(plan_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    events: dict[str, list[tuple[float, int]]] = {}
    for row in plan_rows:
        start = _to_float(row.get("Start Time (min)") or row.get("Start Time") or row.get("Start time (min)"))
        end = _to_float(row.get("End Time (min)") or row.get("End Time") or row.get("End time (min)"))
        if end < start:
            start, end = end, start
        for resource, amount in _parse_required_resources(row.get("Required Resources", "")).items():
            events.setdefault(resource, []).extend([(start, amount), (end, -amount)])

    peaks = {}
    for resource, resource_events in events.items():
        current = peak = 0
        peak_times = []
        for timestamp, delta in sorted(resource_events, key=lambda item: (item[0], item[1])):
            current += delta
            if current > peak:
                peak, peak_times = current, [timestamp]
            elif current == peak and peak > 0:
                peak_times.append(timestamp)
        peaks[resource] = {
            "peak_simultaneous_usage": peak,
            "time_points_reaching_peak_min": sorted(set(peak_times)),
        }
    return peaks


def _calculate_operation_standard_cost(operation_name, operations, resources, requirements):
    if operation_name not in operations:
        return None
    duration = _to_float(operations[operation_name].get("duration_min"))
    required = requirements.get(operation_name, {})
    if not required:
        return 0.0
    total = 0.0
    for resource_name, amount in required.items():
        resource = resources.get(resource_name)
        if resource:
            total += _to_float(resource.get("cost_hour")) * amount * duration / 60.0
    return round(total, 2)


def _cost_within_tolerance(given: float, expected: float, rel_tol: float = 0.0001) -> bool:
    return abs(given - expected) <= 1e-6 if expected == 0 else abs(given - expected) / abs(expected) <= rel_tol


def _normalize_resource_dict(resources_dict: dict[str, int]) -> dict[str, int]:
    normalized = {}
    for name, amount in resources_dict.items():
        canonical = _canonical_resource_name(name)
        normalized[canonical] = normalized.get(canonical, 0) + int(_to_float(amount))
    return normalized


def check_operation_durations(plan_rows, operations):
    incorrect = []
    for row in plan_rows:
        order, operation = row.get("Order", ""), row.get("Operation", "")
        plan_duration = _to_float(row.get("Duration (min)") or row.get("Duration") or row.get("duration"))
        if operation not in operations:
            incorrect.append({
                "order": order,
                "operation": operation,
                "issue": "operation_not_found_in_kg",
                "plan_duration_min": round(plan_duration, 4),
                "correct_duration_min": None,
            })
            continue
        correct = _to_float(operations[operation].get("duration_min"))
        if abs(plan_duration - correct) > 1e-6:
            incorrect.append({
                "order": order,
                "operation": operation,
                "plan_duration_min": round(plan_duration, 4),
                "correct_duration_min": round(correct, 4),
            })
    return {"all_operation_durations_correct": not incorrect, "incorrect_operation_durations": incorrect}


def check_operation_resources(plan_rows, requirements):
    incorrect = []
    for row in plan_rows:
        order, operation = row.get("Order", ""), row.get("Operation", "")
        plan_resources = _parse_required_resources(row.get("Required Resources", ""))
        expected = _normalize_resource_dict(requirements.get(operation, {}))
        if operation not in requirements:
            incorrect.append({
                "order": order,
                "operation": operation,
                "issue": "operation_not_found_in_kg_requirements",
                "plan_resources": plan_resources,
                "correct_resources": None,
            })
            continue
        if plan_resources != expected:
            incorrect.append({
                "order": order,
                "operation": operation,
                "missing_resources": {key: value for key, value in expected.items() if key not in plan_resources},
                "extra_resources": {key: value for key, value in plan_resources.items() if key not in expected},
                "wrong_quantities": {
                    key: {"plan_quantity": plan_resources.get(key), "correct_quantity": expected.get(key)}
                    for key in expected.keys() & plan_resources.keys()
                    if plan_resources.get(key) != expected.get(key)
                },
                "plan_resources": plan_resources,
                "correct_resources": expected,
            })
    return {"all_operation_resources_correct": not incorrect, "incorrect_operation_resources": incorrect}


def check_operation_costs(plan_rows, operations, resources, requirements):
    incorrect = []
    for row in plan_rows:
        order, operation = row.get("Order", ""), row.get("Operation", "")
        given = _to_float(row.get("Cost (€)") or row.get("Cost") or row.get("Cost(EUR)"))
        expected = _calculate_operation_standard_cost(operation, operations, resources, requirements)
        if expected is None:
            incorrect.append({
                "order": order,
                "operation": operation,
                "issue": "operation_not_found_in_kg",
                "given_cost": round(given, 2),
                "correct_cost": None,
            })
        elif not _cost_within_tolerance(given, expected):
            incorrect.append({
                "order": order,
                "operation": operation,
                "given_cost": round(given, 2),
                "correct_cost": round(expected, 2),
            })
    return {"all_operation_costs_correct": not incorrect, "incorrect_operation_costs": incorrect}


def check_resource_peak_usage(plan_rows, resources):
    exceeded = []
    for resource_name, peak_info in sorted(_compute_resource_peaks(plan_rows).items()):
        available = resources.get(resource_name, {}).get("number")
        peak = int(_to_float(peak_info.get("peak_simultaneous_usage")))
        available_number = int(_to_float(available)) if available is not None else None
        if available_number is not None and peak > available_number:
            exceeded.append({
                "resource": resource_name,
                "peak_simultaneous_usage": peak,
                "available_quantity": available_number,
                "exceed_amount": peak - available_number,
            })
    return {"resource_usage_exceeds_limit": bool(exceeded), "exceeded_resources": exceeded}

