"""Gradio application composition and event wiring."""

from __future__ import annotations

import gradio as gr

from ..config import PATHS, SETTINGS
from ..prompts import PLAN_EXAMPLE
from .callbacks import (
    collapse_chat,
    expand_chat,
    handle_chat,
    handle_fullscreen_chat,
    mbse_action,
    monitored_regeneration_action,
    simulation_action,
    verification_action,
)
from .styles import CUSTOM_CSS, GRAPH_PLACEHOLDER_HTML


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        title_row = gr.Row()
        with title_row:
            with gr.Column(scale=1, min_width=120):
                gr.Image(
                    value=str(PATHS.logo),
                    show_label=False,
                    container=False,
                    show_download_button=False,
                    show_share_button=False,
                    interactive=False,
                    show_fullscreen_button=False,
                )
            with gr.Column(scale=2):
                gr.Markdown(
                    """
                    <h1 style="margin-bottom: 2px; font-size: 30px;">
                        A Large Language Model and Knowledge Graph Collaborative Generative Framework for Aircraft Manufacturing System Design in MBSE
                    </h1>
                    <p style="font-size: 24px; margin-top: 0;">
                        Supported by the AI4DESE Laboratory, SUSTech, Shenzhen, China.
                    </p>
                    """,
                    elem_id="custom_title",
                    container=False,
                )

        main_content_row = gr.Row(elem_id="main_content_row")
        with main_content_row:
            with gr.Column(scale=4, elem_id="graph_col"):
                graph_html = gr.HTML(GRAPH_PLACEHOLDER_HTML)
            with gr.Column(scale=5, elem_id="chat_col", elem_classes=["chat-container"]):
                expand_btn = gr.Button("🔍", elem_classes=["expand-btn-embedded"])
                chatbot = gr.Chatbot(label="Chat", elem_id="chatbot", type="messages", height=447)
                user_input = gr.Textbox(
                    placeholder="Ask something...",
                    label=None,
                    lines=2,
                    max_lines=2,
                    elem_id="user_input",
                )
                with gr.Row(elem_id="chat_button_row"):
                    send_btn = gr.Button("Send")
                    clear_btn = gr.Button("Clear")

        examples_row = gr.Row()
        with examples_row:
            process_btn = gr.Button("Process", min_width=50)
            operation_btn = gr.Button("Operation", min_width=100)
            resource_btn = gr.Button("Resource", min_width=50)
            reqresource_btn = gr.Button("ReqResource")
            predecessor_btn = gr.Button("Predecessor")
            plan_btn = gr.Button("Plan", min_width=50)
            mbse_btn = gr.Button("MBSE", min_width=50)
            simulation_btn = gr.Button("Simulation", min_width=150)
            verification_btn = gr.Button("Verification", min_width=150)
            regeneration_btn = gr.Button("Regeneration", min_width=170)

        fullscreen_header = gr.Row(visible=False)
        with fullscreen_header:
            with gr.Column():
                gr.Markdown("## Chat - Fullscreen Mode", elem_classes=["fullscreen-title"])
            with gr.Column(scale=0):
                collapse_btn = gr.Button("✖ Close Fullscreen", elem_classes=["close-btn"])

        fullscreen_chatbot = gr.Chatbot(
            label=None,
            elem_classes=["fullscreen-chat"],
            type="messages",
            height=600,
            visible=False,
        )
        fullscreen_input_row = gr.Row(visible=False)
        with fullscreen_input_row:
            with gr.Column():
                fullscreen_user_input = gr.Textbox(placeholder="Ask something...", label=None, lines=2)
                with gr.Row():
                    fullscreen_send_btn = gr.Button("Send")
                    fullscreen_clear_btn = gr.Button("Clear")

        send_btn.click(handle_chat, [user_input, chatbot], [chatbot, graph_html])
        clear_btn.click(lambda: ([], GRAPH_PLACEHOLDER_HTML), None, [chatbot, graph_html])
        fullscreen_send_btn.click(
            handle_fullscreen_chat,
            [fullscreen_user_input, fullscreen_chatbot],
            fullscreen_chatbot,
        )
        fullscreen_clear_btn.click(lambda: [], None, fullscreen_chatbot)

        fullscreen_outputs = [
            title_row,
            main_content_row,
            examples_row,
            fullscreen_header,
            fullscreen_chatbot,
            fullscreen_input_row,
        ]
        expand_btn.click(expand_chat, chatbot, fullscreen_outputs + [fullscreen_chatbot])
        collapse_btn.click(collapse_chat, fullscreen_chatbot, fullscreen_outputs + [chatbot])

        user_input.submit(handle_chat, [user_input, chatbot], [chatbot, graph_html])
        fullscreen_user_input.submit(
            handle_fullscreen_chat,
            [fullscreen_user_input, fullscreen_chatbot],
            fullscreen_chatbot,
        )

        process_btn.click(lambda: "List all information of processes and their sub-processes.", None, user_input)
        operation_btn.click(lambda: "List all information of operations.", None, user_input)
        resource_btn.click(lambda: "List all information of resources.", None, user_input)
        reqresource_btn.click(
            lambda: "Search all relationships between operations and resources. List all names of operations, names of resources, and number of need resources. Merge information according to the operation.",
            None,
            user_input,
        )
        predecessor_btn.click(lambda: "List all predecessors of each operation.", None, user_input)
        plan_btn.click(lambda: PLAN_EXAMPLE, None, user_input)
        mbse_btn.click(mbse_action, chatbot, chatbot)
        simulation_btn.click(simulation_action, chatbot, chatbot)
        verification_btn.click(verification_action, chatbot, chatbot)
        regeneration_btn.click(monitored_regeneration_action, [user_input, chatbot], chatbot)

    return demo


def launch() -> None:
    PATHS.static.mkdir(parents=True, exist_ok=True)
    build_demo().queue().launch(server_name=SETTINGS.host, server_port=SETTINGS.port)

