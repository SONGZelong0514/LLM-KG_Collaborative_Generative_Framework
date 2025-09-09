import gradio as gr
from backend import smart_qa_system
import sys
import os
import time
from pathlib import Path
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import csv2GOPPRRE
import GOPPRRE2sim

custom_css = """
* {
    font-family: 'Times New Roman', Times, serif !important;
}

button {
    font-size: 24px !important;
    font-weight: bold !important;
    font-family: 'Times New Roman', Times, serif !important;
}

.prose, .markdown {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

.gradio-textbox input, .gradio-textbox textarea {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

.gradio-textbox input, 
.gradio-textbox textarea,
input[type="text"],
textarea,
.gr-textbox input,
.gr-textbox textarea {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

input::placeholder,
textarea::placeholder {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

input:focus,
textarea:focus,
.gradio-textbox input:focus,
.gradio-textbox textarea:focus {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

div[data-testid="textbox"] input,
div[data-testid="textbox"] textarea {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

.chatbot {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Times New Roman', Times, serif !important;
}

.gradio-container, .main, body {
    background-color: white !important;
}

.chat-container {
    position: relative !important;
}

.expand-btn-embedded {
    position: absolute !important;
    top: 8px !important;
    right: 8px !important;
    z-index: 100 !important;
    font-size: 12px !important;
    min-width: 22px !important;
    max-width: 22px !important;
    height: 22px !important;
    padding: 0px !important;
    background-color: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid #ddd !important;
    border-radius: 3px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    opacity: 0.7 !important;
    transition: all 0.2s ease !important;
}

.expand-btn-embedded:hover {
    background-color: rgba(240, 240, 240, 0.95) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
    opacity: 1 !important;
    transform: scale(1.05) !important;
}

.fullscreen-chat {
    font-size: 28px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

.close-btn {
    background-color: #ff4757 !important;
    color: white !important;
    font-size: 18px !important;
    min-width: 100px !important;
    margin-bottom: 10px !important;
}

.fullscreen-title {
    font-size: 24px !important;
    font-weight: bold !important;
    margin-bottom: 10px !important;
    color: #333 !important;
}
"""

GRAPH_PLACEHOLDER_HTML = (
    "<div style='border: 1px solid #ccc; padding: 10px; height: 650px;"
    "display: flex; align-items: center; justify-content: center;'>"
    "<p style='color: #666;'>Graph will be shown here after querying...</p>"
    "</div>"
)

def clean_old_graphs(static_folder="static", max_age=3600):
    now = time.time()
    p = Path(static_folder)
    if not p.exists():
        return
    for f in p.glob("graph_*.html"):
        if now - f.stat().st_mtime > max_age:
            try:
                f.unlink()
            except Exception as e:
                print("Delete graph failed:", e)

def get_graph_html_content(graph_html_path):
    if not graph_html_path:
        return "There are no graph data."
    full = Path(os.getcwd()) / graph_html_path.lstrip("/")
    if not full.exists():
        return "There are no graph data."
    html = full.read_text(encoding="utf-8")
    return f"""
    <div style='width: 100%; height: 650px; border: 1px solid #ccc; overflow: hidden;'>
        <iframe srcdoc="{html.replace('"', '&quot;')}"
                style="width: 100%; height: 100%; border: none;"></iframe>
    </div>"""

def mbse_action(history):
    plans_dir = './plans'
    mbse_dir = './MBSE'
    os.makedirs(mbse_dir, exist_ok=True)

    csv_files = [
        f for f in os.listdir(plans_dir)
        if f.startswith('assembly_plan_') and f.endswith('.csv')
    ]
    if not csv_files:
        return history + [{"role": "assistant", "content":
            "❌ No assembly_plan_*.csv file found in ./plans. Please generate the assembly plan first!"}]

    csv_files.sort()
    latest_csv = csv_files[-1]
    latest_csv_path = os.path.join(plans_dir, latest_csv)
    time_suffix = latest_csv[len('assembly_plan_'):-len('.csv')]

    frag_txt = os.path.join(mbse_dir, f"owl_out_{time_suffix}.txt")
    merged_owl = os.path.join(mbse_dir, f"assembly_plan_MBSE_{time_suffix}.owl")
    main_owl = os.path.join(current_dir, "GOPPRRE.owl")

    csv2GOPPRRE.build(latest_csv_path, frag_txt)
    csv2GOPPRRE.merge_fragment(main_owl, frag_txt, merged_owl)

    if os.path.exists(frag_txt):
        os.remove(frag_txt)

    msg = (
        "✅ The MBSE model of the assembly plan has been saved as an OWL file:\n"
        f"./MBSE/assembly_plan_MBSE_{time_suffix}.owl"
    )
    return history + [{"role": "assistant", "content": msg}]

def simulation_action(history):
    sim_dir = './Simulation'
    mbse_dir = './MBSE'
    os.makedirs(sim_dir, exist_ok=True)

    owl_files = [
        f for f in os.listdir(mbse_dir)
        if f.startswith('assembly_plan_MBSE_') and f.endswith('.owl')
    ]
    if not owl_files:
        return history + [{"role": "assistant", "content":
            "❌ No assembly_plan_MBSE_*.owl file found in ./MBSE. Please generate the MBSE model first!"}]

    owl_files.sort()
    latest_owl = owl_files[-1]
    latest_owl_path = os.path.join(mbse_dir, latest_owl)
    m_file_name = latest_owl.replace('.owl', '.m')
    m_file_path = os.path.join(sim_dir, m_file_name)

    try:
        GOPPRRE2sim.owl_to_matlab(latest_owl_path, m_file_path)
        msg = f"✅ The MATLAB simulation file has been generated:\n{m_file_path}"
    except Exception as e:
        msg = f"❌ Simulation failed: {e}"

    return history + [{"role": "assistant", "content": msg}]

with gr.Blocks(css=custom_css) as demo:
    title_row = gr.Row()
    with title_row:
        with gr.Column(scale=1, min_width=120):
            gr.Image(value="logo.png", show_label=False, container=False, show_download_button=False,
                     show_share_button=False, interactive=False, show_fullscreen_button=False)
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
                container=False
            )

    main_content_row = gr.Row()
    with main_content_row:
        with gr.Column(scale=4):
            graph_html = gr.HTML(GRAPH_PLACEHOLDER_HTML)

        with gr.Column(scale=5, elem_classes=["chat-container"]):
            expand_btn = gr.Button("🔍", elem_classes=["expand-btn-embedded"])

            chatbot = gr.Chatbot(label="Chat", elem_id="chatbot", type="messages", height=447)
            user_input = gr.Textbox(placeholder="Ask something...", label=None, lines=1)

            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear")

    examples_row = gr.Row()
    with examples_row:
        gr.Markdown("**Examples:**")
        process_btn = gr.Button("Process", min_width=80)
        operation_btn = gr.Button("Operation")
        resource_btn = gr.Button("Resource")
        reqresource_btn = gr.Button("ReqResource")
        predecessor_btn = gr.Button("Predecessor")
        plan_btn = gr.Button("Plan", min_width=80)
        mbse_btn = gr.Button("MBSE", min_width=80)
        simulation_btn = gr.Button("Simulation")

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
        visible=False
    )

    fullscreen_input_row = gr.Row(visible=False)
    with fullscreen_input_row:
        with gr.Column():
            fullscreen_user_input = gr.Textbox(placeholder="Ask something...", label=None, lines=2)
            with gr.Row():
                fullscreen_send_btn = gr.Button("Send")
                fullscreen_clear_btn = gr.Button("Clear")

    def handle_chat(user_msg, history):
        clean_old_graphs()
        assistant_partial = ""
        graph_html_content = GRAPH_PLACEHOLDER_HTML

        for part, g_path in smart_qa_system(user_msg):
            if g_path and (graph_html_content == GRAPH_PLACEHOLDER_HTML or not graph_html_content):
                graph_html_content = get_graph_html_content(g_path)

            assistant_partial += part

            yield (
                history +
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_partial}],
                graph_html_content
            )


    def handle_fullscreen_chat(user_msg, history):
        assistant_partial = ""

        for part, g_path in smart_qa_system(user_msg):
            # 累加回答
            assistant_partial += part

            # 输出给前端
            yield (
                    history +
                    [{"role": "user", "content": user_msg},
                     {"role": "assistant", "content": assistant_partial}]
            )


    def expand_chat(chatbot_history):
        return [
            gr.update(visible=False),  # title_row
            gr.update(visible=False),  # main_content_row
            gr.update(visible=False),  # examples_row
            gr.update(visible=True),  # fullscreen_header
            gr.update(visible=True),  # fullscreen_chatbot
            gr.update(visible=True),  # fullscreen_input_row
            chatbot_history  # sync chatbot history
        ]


    def collapse_chat(fullscreen_chatbot_history):
        return [
            gr.update(visible=True),  # title_row
            gr.update(visible=True),  # main_content_row
            gr.update(visible=True),  # examples_row
            gr.update(visible=False),  # fullscreen_header
            gr.update(visible=False),  # fullscreen_chatbot
            gr.update(visible=False),  # fullscreen_input_row
            fullscreen_chatbot_history  # sync chatbot history back
        ]

    send_btn.click(
        fn=handle_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, graph_html],
    )

    clear_btn.click(
        fn=lambda: ([], GRAPH_PLACEHOLDER_HTML),
        inputs=[],
        outputs=[chatbot, graph_html]
    )

    fullscreen_send_btn.click(
        fn=handle_fullscreen_chat,
        inputs=[fullscreen_user_input, fullscreen_chatbot],
        outputs=[fullscreen_chatbot],
    )

    fullscreen_clear_btn.click(
        fn=lambda: [],
        inputs=[],
        outputs=[fullscreen_chatbot]
    )

    expand_btn.click(
        fn=expand_chat,
        inputs=[chatbot],
        outputs=[title_row, main_content_row, examples_row, fullscreen_header, fullscreen_chatbot, fullscreen_input_row,
                 fullscreen_chatbot]
    )

    collapse_btn.click(
        fn=collapse_chat,
        inputs=[fullscreen_chatbot],
        outputs=[title_row, main_content_row, examples_row, fullscreen_header, fullscreen_chatbot, fullscreen_input_row,
                 chatbot]
    )

    user_input.submit(
        fn=handle_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, graph_html],
    )

    fullscreen_user_input.submit(
        fn=handle_fullscreen_chat,
        inputs=[fullscreen_user_input, fullscreen_chatbot],
        outputs=[fullscreen_chatbot],
    )

    process_btn.click(
        lambda: "List all information of processes and their sub-processes.", None, user_input)
    operation_btn.click(
        lambda: "List all information of operations.", None, user_input)
    resource_btn.click(
        lambda: "List all information of resources.", None, user_input)
    reqresource_btn.click(
        lambda: "Search all relationships between operations and resources. List all names of operations, names of resources, and number of need resources. Merge information according to the operation.",
        None, user_input)
    predecessor_btn.click(
        lambda: "List all predecessors of each operation.", None, user_input)
    plan_btn.click(
        lambda: """Please help me design a complete aircraft fuselage joint plan that includes four 1/4 bodies, using both automatic and manual methods. Your plan should follow these specific constraints:

1. Each operation depends on preceding operations, which must be completed first.
2. The first two operations must be "S40_00001_Jig in" and "S40_01001_Set up working environment", and the last operation must be "S40_00002_Jig out". They only need to be executed once during the whole joint plan.
3. "S40_04012_Deburring int, positioning, attach them automatic" or "S40_04013_Deburring int, positioning, attach them manual" marks the completion of one 1/4 aircraft fuselage joint. You must include a total of four such operations to complete the joint of the entire aircraft fuselage.
4. Automatic joint operations can only be performed in series, and only one 1/4 body can be automatically jointed at a time. However, manual joint can be done in series or parallel. At most two sets of manual operations can be carried out in parallel.
5. Manual and automatic operations cannot be carried out in parallel.
6. Only the same manual operations can be carried out in parallel. Different manual operations cannot be carried out in parallel.
7. After "S40_04014_Deinstall LFT and rails" and "S40_04013_Deburring int, positioning, attach them manual", the next two steps must be: "S40_02002_Cleanup and add sealant" then "S40_02003_Inspection".
8. If multiple 1/4 bodies are manually jointed in series or parallel, it only needs to execute "S40_02002_Cleanup and add sealant" and "S40_02003_Inspection" at the final completion.
9. If multiple 1/4 bodies are automatically jointed in series, only "S40_02001_Set in position Rails and LFT" needs to be executed first, and "S40_04014-Deinstall LFT and rails" needs to be executed last.""",
        None, user_input)

    mbse_btn.click(
        mbse_action,
        inputs=[chatbot],
        outputs=chatbot
    )

    simulation_btn.click(
        simulation_action,
        inputs=[chatbot],
        outputs=chatbot
    )

# ------------------------- 启动 ----------------------------
if __name__ == "__main__":
    static_dir = os.path.join(os.getcwd(), "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    demo.queue().launch(server_name="localhost",
                        server_port=7860)