"""Knowledge-graph question answering and assembly-plan generation."""

from __future__ import annotations

import datetime
import io
import re
from contextlib import redirect_stdout
from functools import lru_cache
from typing import Generator, Optional

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

from .clients import get_chat_model, get_graph, get_openai_client
from .config import PATHS, SETTINGS
from .graph_visualization import generate_graph_html
from .plans import save_table_from_response
from .prompts import (
    DESIGN_QA_PROMPT_TEMPLATE,
    GRAPH_RESPONSE_PROMPT_TEMPLATE,
    RETRY_PREFIX_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
)
from .streaming import extract_stream_token


EXCLUDED_GRAPH_TYPES = [
    "Class",
    "Relationship",
    "_GraphConfig",
    "SCO_RESTRICTION",
    "DOMAIN",
    "RANGE",
    "isSubClassOf",
    "isSubPropertyOf",
    "hasOptionalAutoOperation",
    "hasOptionalManualOperation",
]

FIXED_KG_QUERIES = [
    ("processes", "MATCH (p:Process)-[:hasSubprocess]->(sp:Process)\nRETURN p, sp"),
    ("operations", "MATCH (o:Operation)\nRETURN o"),
    ("resources", "MATCH (r:Resource)\nRETURN r"),
    (
        "operation_resource_dependencies",
        "MATCH (o:Operation)-[r:requiresResource]->(res:Resource)\n"
        "RETURN o.name AS operationName, res.name AS resourceName, r.number AS requiredResources",
    ),
    (
        "operation_predecessor_dependencies",
        "MATCH (o:Operation)-[:hasPredecessors]->(p:Operation)\n"
        "RETURN o.name AS Operation, collect(p.name) AS Predecessors",
    ),
]


@lru_cache(maxsize=1)
def _router_chain():
    prompt = PromptTemplate(input_variables=["question"], template=ROUTER_PROMPT_TEMPLATE)
    return prompt | get_chat_model(streaming=False)


@lru_cache(maxsize=1)
def _cypher_chain():
    return GraphCypherQAChain.from_llm(
        llm=get_chat_model(streaming=False),
        graph=get_graph(),
        allow_dangerous_requests=True,
        verbose=True,
        exclude_types=EXCLUDED_GRAPH_TYPES,
        top_k=300,
        return_direct=True,
        return_intermediate_steps=True,
    )


def normalize_router_output(raw_text: str) -> str:
    if not raw_text:
        return "graph"
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text.strip().lower(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    matches = re.findall(r"\b(graph|design)\b", text)
    return matches[-1] if matches else "graph"


def clean_cypher_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:cypher)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    return re.sub(r"^\s*cypher\s*", "", text, flags=re.IGNORECASE).strip()


def extract_generated_cypher_from_text(text: str) -> str:
    marker = "Generated Cypher:"
    if not text or marker not in text:
        return ""
    cypher_text = text[text.find(marker) + len(marker) :].strip()
    for stop_marker in ("Full Context:", "Error:"):
        position = cypher_text.find(stop_marker)
        if position != -1:
            cypher_text = cypher_text[:position].strip()
    return clean_cypher_text(cypher_text)


def escape_prompt_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}") if text else ""


def invoke_chain_and_capture_stdout(chain, question: str):
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            result = chain.invoke({"query": question})
        return True, result, buffer.getvalue()
    except Exception as exc:
        return False, exc, buffer.getvalue()


def build_retry_cypher_chain(bad_cypher: str, error_message: str):
    from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT

    retry_prefix = RETRY_PREFIX_TEMPLATE.format(
        bad_cypher=escape_prompt_braces(bad_cypher),
        error_message=escape_prompt_braces(error_message),
    )
    retry_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=retry_prefix + CYPHER_GENERATION_PROMPT.template,
    )
    return GraphCypherQAChain.from_llm(
        llm=get_chat_model(streaming=False),
        graph=get_graph(),
        allow_dangerous_requests=True,
        verbose=True,
        exclude_types=EXCLUDED_GRAPH_TYPES,
        top_k=300,
        cypher_prompt=retry_prompt,
        return_direct=True,
        return_intermediate_steps=True,
    )


def run_cypher_with_retry(question: str):
    success, first_result, first_stdout = invoke_chain_and_capture_stdout(_cypher_chain(), question)
    if success:
        graph_data = first_result.get("result", first_result)
        cypher = first_result.get("intermediate_steps", [{}])[0].get("query", "")
        return graph_data, clean_cypher_text(cypher)

    first_error = str(first_result)
    bad_cypher = extract_generated_cypher_from_text(first_stdout)
    if not bad_cypher:
        raise RuntimeError(f"无法从第一次执行日志中提取 Cypher。\n原始错误：{first_error}") from first_result

    retry_chain = build_retry_cypher_chain(bad_cypher, first_error)
    success, second_result, second_stdout = invoke_chain_and_capture_stdout(retry_chain, question)
    if success:
        graph_data = second_result.get("result", second_result)
        cypher = second_result.get("intermediate_steps", [{}])[0].get("query", "")
        return graph_data, clean_cypher_text(cypher)

    second_error = str(second_result)
    second_cypher = extract_generated_cypher_from_text(second_stdout)
    raise RuntimeError(
        "Cypher 查询失败（已重试一次）。\n\n"
        f"第一次 Cypher:\n{bad_cypher}\n\n第一次错误:\n{first_error}\n\n"
        f"第二次 Cypher:\n{second_cypher}\n\n第二次错误:\n{second_error}"
    ) from second_result


def build_design_kg_context():
    context_blocks = []
    retrieved_pairs = []
    graph = get_graph()
    for name, cypher in FIXED_KG_QUERIES:
        result = graph.query(cypher)
        retrieved_pairs.append({"name": name, "cypher": cypher, "result": result})
        context_blocks.append(
            f"[Knowledge Block: {name}]\nCypher:\n{cypher}\nGraph Data:\n{result}\n"
        )
    return "\n\n".join(context_blocks), retrieved_pairs


def graph_answer_token_stream(question: str, graph_data, cypher: str):
    prompt = PromptTemplate(
        input_variables=["question", "graph_data", "cypher"],
        template=GRAPH_RESPONSE_PROMPT_TEMPLATE,
    ).format(question=question, graph_data=graph_data, cypher=cypher)
    stream = get_openai_client().chat.completions.create(
        model=SETTINGS.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=SETTINGS.temperature,
        stream=True,
    )
    for chunk in stream:
        token = extract_stream_token(chunk, hide_thinking=True)
        if token:
            print(token, end="", flush=True)
            yield token


def design_answer_token_stream(question: str, kg_context: str):
    prompt = PromptTemplate(
        input_variables=["question", "kg_context"],
        template=DESIGN_QA_PROMPT_TEMPLATE,
    ).format(question=question, kg_context=kg_context)
    stream = get_openai_client().chat.completions.create(
        model=SETTINGS.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=SETTINGS.temperature,
        stream=True,
    )
    for chunk in stream:
        token = extract_stream_token(chunk, hide_thinking=True)
        if token:
            print(token, end="", flush=True)
            yield token


def smart_qa_system(question: str) -> Generator[tuple[str, Optional[str]], None, None]:
    graph_html_path = None
    answer_full = ""
    try:
        raw_response = _router_chain().invoke({"question": question}).content
        response_type = normalize_router_output(raw_response)

        if response_type == "graph":
            graph_data, cypher = run_cypher_with_retry(question)
            graph_html_path = generate_graph_html(graph_data)
            yield "", graph_html_path
            for token in graph_answer_token_stream(question, graph_data, cypher):
                answer_full += token
                yield token, graph_html_path
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            PATHS.constraints.mkdir(parents=True, exist_ok=True)
            constraint_path = PATHS.constraints / f"assembly_plan_design_constraint_{timestamp}.txt"
            constraint_path.write_text(question, encoding="utf-8")
            print(f"【系统】设计约束已保存: {constraint_path}")
        except Exception as exc:
            print(f"【警告】约束保存失败: {exc}")
        yield "", None

        kg_context, retrieved_pairs = build_design_kg_context()
        for item in retrieved_pairs:
            print(f"- {item['name']}: {item['cypher']}")
            print(f"  返回记录数: {len(item['result']) if isinstance(item['result'], list) else 'N/A'}")

        for token in design_answer_token_stream(question, kg_context):
            answer_full += token
            yield token, None

        csv_message = ""
        try:
            csv_path = PATHS.plans / f"assembly_plan_{timestamp}.csv"
            save_table_from_response(answer_full, csv_path)
            csv_message = f"\n\n✅ **The assembly plan has been saved as a CSV file:** `./plans/{csv_path.name}`"
        except Exception as exc:
            print(f"【错误】CSV 生成失败: {exc}")
            csv_message = "\n⚠️ No formatting compliant Markdown table detected, CSV not saved."

        answer_full += csv_message
        yield csv_message, None
    except Exception as exc:
        print("【错误】", exc)
        yield f"抱歉，处理您的问题时出错：{exc}", None

