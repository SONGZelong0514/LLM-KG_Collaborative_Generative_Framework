"""Visual styles and static HTML used by the unchanged Gradio interface."""

CUSTOM_CSS = """
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

.gradio-textbox input, .gradio-textbox textarea,
input[type="text"], textarea,
.gr-textbox input, .gr-textbox textarea {
    font-size: 24px !important;
    font-family: 'Times New Roman', Times, serif !important;
}

input::placeholder, textarea::placeholder,
input:focus, textarea:focus,
.gradio-textbox input:focus, .gradio-textbox textarea:focus,
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

#main_content_row {
    align-items: flex-start !important;
}

#graph_col, #chat_col {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

#chat_col {
    height: 650px !important;
    display: flex !important;
    flex-direction: column !important;
}

#chatbot {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

#user_input {
    flex: 0 0 auto !important;
}

#user_input textarea {
    height: 60px !important;
    max-height: 60px !important;
    overflow-y: auto !important;
    resize: none !important;
}

#chat_button_row {
    flex: 0 0 auto !important;
}
"""


GRAPH_PLACEHOLDER_HTML = (
    "<div style='border: 1px solid #ccc; padding: 10px; height: 650px;"
    "display: flex; align-items: center; justify-content: center;'>"
    "<p style='color: #666;'>Graph will be shown here after querying...</p>"
    "</div>"
)

