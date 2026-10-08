"""Convert Neo4j query results into embeddable PyVis HTML."""

from __future__ import annotations

import uuid
from pathlib import Path

from pyvis.network import Network

from .config import PATHS


def add_node_if_absent(net, node_records, node_id, label=None, color="#97c2fc", shape="box"):
    if node_id not in node_records:
        net.add_node(node_id, label=label or node_id, color=color, shape=shape)
        node_records[node_id] = True


def generate_attribute_label(node_data):
    name = node_data.get("name", "Node")
    attributes = "\n".join(f"{key}: {value}" for key, value in node_data.items() if key != "name")
    return f"{name}\n{attributes}" if attributes else name


def detect_data_format(data):
    if not data or not isinstance(data, list) or not isinstance(data[0], dict):
        return "A"
    sample = data[0]
    dict_fields = [key for key, value in sample.items() if isinstance(value, dict)]
    if len(dict_fields) == 1 and len(sample) >= 3:
        return "C"
    if not dict_fields:
        if len(sample) == 2:
            return "B"
        if len(sample) >= 3:
            return "D"
    return "A"


def process_format_a(net, data, node_records):
    for item in data:
        if not isinstance(item, dict):
            continue
        for value in item.values():
            if isinstance(value, dict):
                node_id = str(value.get("name", uuid.uuid4()))
                add_node_if_absent(net, node_records, node_id, label=generate_attribute_label(value))


def process_format_b(net, data, node_records):
    if not data or not isinstance(data[0], dict):
        return
    keys = list(data[0].keys())[:2]
    if len(keys) < 2:
        return
    field1, field2 = keys
    for item in data:
        node1 = str(item.get(field1, ""))
        node2 = str(item.get(field2, ""))
        if node1 and node2:
            add_node_if_absent(net, node_records, node1, color="#97c2fc", shape="box")
            add_node_if_absent(net, node_records, node2, color="#fc9797", shape="box")
            net.add_edge(node1, node2, color="#666666")


def process_format_c(net, data, node_records):
    if not data or not isinstance(data[0], dict):
        return
    sample = data[0]
    dict_field = next((key for key, value in sample.items() if isinstance(value, dict)), None)
    if not dict_field:
        return

    for item in data:
        dict_data = item.get(dict_field, {})
        node1 = str(dict_data.get("name", uuid.uuid4()))
        add_node_if_absent(net, node_records, node1, label=generate_attribute_label(dict_data))
        other_keys = [key for key in item if key != dict_field]
        if not other_keys:
            continue
        node2_field = other_keys[0]
        node2 = str(item.get(node2_field, ""))
        add_node_if_absent(net, node_records, node2, color="#fc9797", shape="diamond")
        relation_field = next((key for key in other_keys if key != node2_field), None)
        relation = str(item.get(relation_field, "")) if relation_field else ""
        net.add_edge(node1, node2, label=relation, color="#666666")


def process_format_d(net, data, node_records):
    if not data or not isinstance(data[0], dict):
        return
    fields = list(data[0].keys())
    if len(fields) < 3:
        return
    node1_field, node2_field, relation_field = fields[:3]
    for item in data:
        node1 = str(item.get(node1_field, ""))
        node2 = str(item.get(node2_field, ""))
        relation = str(item.get(relation_field, ""))
        if node1 and node2:
            add_node_if_absent(net, node_records, node1)
            add_node_if_absent(net, node_records, node2)
            net.add_edge(node1, node2, label=relation, color="#666666")


def configure_network(net):
    net.toggle_physics(False)
    net.set_options(
        """
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        },
        "nodes": {
            "font": {
                "size": 14
            }
        }
    }
    """
    )


def save_network(net) -> str:
    PATHS.static.mkdir(parents=True, exist_ok=True)
    filename = f"graph_{uuid.uuid4().hex}.html"
    # PyVis writes with the platform default encoding on Windows.  Knowledge-
    # graph properties may contain characters that GBK cannot represent, so
    # generate the document in memory and persist it explicitly as UTF-8.
    html = net.generate_html()
    (PATHS.static / filename).write_text(html, encoding="utf-8")
    return f"/static/{filename}"


def generate_graph_html(graph_data) -> str:
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    node_records = {}
    handlers = {
        "A": process_format_a,
        "B": process_format_b,
        "C": process_format_c,
        "D": process_format_d,
    }
    handlers[detect_data_format(graph_data)](net, graph_data, node_records)
    configure_network(net)
    return save_network(net)


def get_graph_html_content(graph_html_path: str | None) -> str:
    if not graph_html_path:
        return "There are no graph data."
    full_path = PATHS.static / Path(graph_html_path).name
    if not full_path.exists():
        return "There are no graph data."
    html = full_path.read_text(encoding="utf-8")
    return f"""
    <div style='width: 100%; height: 650px; border: 1px solid #ccc; overflow: hidden;'>
        <iframe srcdoc="{html.replace('"', '&quot;')}"
                style="width: 100%; height: 100%; border: none;"></iframe>
    </div>"""


def clean_old_graphs(max_age: int = 3600) -> None:
    import time

    if not PATHS.static.exists():
        return
    now = time.time()
    for graph_file in PATHS.static.glob("graph_*.html"):
        if now - graph_file.stat().st_mtime > max_age:
            try:
                graph_file.unlink()
            except OSError as exc:
                print("Delete graph failed:", exc)

