"""Read verification evidence from the Neo4j knowledge graph."""

from __future__ import annotations

from typing import Any

from ..clients import get_graph
from .checks import _canonical_resource_name, _to_float


def _extract_node_properties(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key, {})
    if isinstance(value, dict):
        properties = value.get("properties")
        return properties if isinstance(properties, dict) else value
    return {}


def query_operation_dependencies() -> dict[str, list[str]]:
    result = get_graph().query(
        "MATCH (o:Operation)-[:hasPredecessors]->(p:Operation)\n"
        "RETURN o.name AS Operation, collect(p.name) AS Predecessors"
    )
    return {
        str(record["Operation"]): [str(value) for value in record.get("Predecessors", [])]
        for record in result
        if record.get("Operation")
    }


def query_resource_knowledge() -> dict[str, dict[str, Any]]:
    resources = {}
    for record in get_graph().query("MATCH (r:Resource)\nRETURN r"):
        properties = _extract_node_properties(record, "r")
        name = properties.get("name")
        if not name:
            continue
        canonical = _canonical_resource_name(name)
        resources[canonical] = {
            "name": str(name),
            "canonical_name": canonical,
            "number": _to_float(properties.get("number")),
            "calendar": properties.get("calendar", ""),
            "cost_hour": _to_float(properties.get("cost_hour")),
        }
    return resources


def query_operation_knowledge() -> dict[str, dict[str, Any]]:
    operations = {}
    for record in get_graph().query("MATCH (o:Operation)\nRETURN o"):
        properties = _extract_node_properties(record, "o")
        name = properties.get("name")
        if name:
            operations[str(name)] = {
                "name": str(name),
                "duration_min": _to_float(properties.get("duration")),
                "op_type": properties.get("op_type", ""),
            }
    return operations


def query_operation_resource_requirements() -> dict[str, dict[str, int]]:
    result = get_graph().query(
        "MATCH (o:Operation)-[r:requiresResource]->(res:Resource)\n"
        "RETURN o.name AS operationName, res.name AS resourceName, r.number AS requiredResources"
    )
    requirements: dict[str, dict[str, int]] = {}
    for record in result:
        operation, resource = record.get("operationName"), record.get("resourceName")
        if not operation or not resource:
            continue
        requirements.setdefault(str(operation), {})[_canonical_resource_name(resource)] = int(
            _to_float(record.get("requiredResources"))
        )
    return requirements

