"""Callbacks used by the Gradio interface."""

from __future__ import annotations

import gradio as gr

from ..graph_visualization import clean_old_graphs, get_graph_html_content
from ..monitoring import RuntimeGpuMonitor, append_metrics
from ..private_plugins import generate_mbse_model, generate_simulation_model
from ..qa import smart_qa_system
from ..regeneration import regeneration_action
from ..verification import run_verification_stream
from .styles import GRAPH_PLACEHOLDER_HTML


def handle_chat(user_msg, history):
    history = history or []
    clean_old_graphs()
    graph_html_content = GRAPH_PLACEHOLDER_HTML
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    yield history, graph_html_content

    monitor = RuntimeGpuMonitor("Initial plan generation LLM").start()
    try:
        for part, graph_path in smart_qa_system(user_msg):
            if graph_path:
                graph_html_content = get_graph_html_content(graph_path)
            if part:
                history[-1]["content"] += str(part)
            yield history, graph_html_content
    finally:
        yield append_metrics(history, monitor), graph_html_content


def handle_fullscreen_chat(user_msg, history):
    history = (history or []) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": ""},
    ]
    yield history
    monitor = RuntimeGpuMonitor("Initial plan generation LLM").start()
    try:
        for part, _ in smart_qa_system(user_msg):
            if part:
                history[-1]["content"] += str(part)
            yield history
    finally:
        yield append_metrics(history, monitor)


def verification_action(history):
    history = (history or []) + [{"role": "assistant", "content": "🔍 Running verification...\n"}]
    yield history
    monitor = RuntimeGpuMonitor("Verification LLM").start()
    try:
        for token in run_verification_stream():
            history[-1]["content"] += token
            yield history
    except Exception as exc:
        history[-1]["content"] += f"\n\n❌ Verification failed: {exc}"
    yield append_metrics(history, monitor)


def monitored_regeneration_action(human_feedback, history):
    monitor = RuntimeGpuMonitor("Regeneration LLM").start()
    last_history = history or []
    try:
        for updated_history in regeneration_action(human_feedback, history):
            last_history = updated_history
            yield updated_history
    finally:
        if last_history:
            yield append_metrics(last_history, monitor)


def mbse_action(history):
    try:
        output = generate_mbse_model()
        message = f"✅ The MBSE model of the assembly plan has been saved as an OWL file:\n./MBSE/{output.name}"
    except Exception as exc:
        message = f"❌ MBSE generation failed: {exc}"
    return (history or []) + [{"role": "assistant", "content": message}]


def simulation_action(history):
    try:
        output = generate_simulation_model()
        message = f"✅ The MATLAB simulation file has been generated:\n./Simulation/{output.name}"
    except Exception as exc:
        message = f"❌ Simulation failed: {exc}"
    return (history or []) + [{"role": "assistant", "content": message}]


def expand_chat(chatbot_history):
    return [
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        chatbot_history,
    ]


def collapse_chat(fullscreen_chatbot_history):
    return [
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        fullscreen_chatbot_history,
    ]

