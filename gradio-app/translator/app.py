import gradio as gr
from backend import backend

# Define languages
LANGUAGES = ["Korean", "English", "Chinese", "Japanese", "Vietnamese"]


def translate_wrapper(text, src_lang, tgt_lang):
    if not text.strip():
        yield ""
        return

    # Use the generator from backend
    result = ""
    for chunk in backend.generate_analysis(text, src_lang, tgt_lang):
        result = chunk
        yield result


custom_css = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
#title-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
    border-radius: 1rem;
}
.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div id="title-header">
            <h1>🌍 AI Polyglot Translator</h1>
            <p>Professional Real-time Translation powered by LLMs</p>
        </div>
        """,
        elem_id="title-header",
    )

    with gr.Row():
        with gr.Column(scale=1):
            src_lang = gr.Radio(
                choices=LANGUAGES,
                value="English",
                label="Source Language",
            )
            src_text = gr.Textbox(
                label="Input Text",
                placeholder="Type or paste text here...",
                lines=8,
                max_lines=15,
            )

        with gr.Column(scale=1):
            tgt_lang = gr.Radio(
                choices=LANGUAGES,
                value="Korean",
                label="Target Language",
            )
            tgt_text = gr.Textbox(
                label="Translated Text",
                placeholder="Translation will appear here...",
                lines=8,
                max_lines=15,
                interactive=False,
            )

    with gr.Row():
        translate_btn = gr.Button(
            "Translate", variant="primary", scale=2, elem_classes=["primary-btn"]
        )
        clear_btn = gr.Button("Clear All", variant="secondary", scale=1)

    with gr.Accordion("Settings & Model Info", open=False):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=backend.available_models,
                value=backend.available_models[0],
                label="Select Engine Model",
            )
            status_msg = gr.Markdown(f"Current Model: {backend.available_models[0]}")

        def update_model(model_name):
            backend.load_model(model_name)
            return f"Current Model: {model_name}"

        model_dropdown.change(
            fn=update_model, inputs=[model_dropdown], outputs=[status_msg]
        )

    # Event handlers
    translate_btn.click(
        fn=translate_wrapper, inputs=[src_text, src_lang, tgt_lang], outputs=[tgt_text]
    )

    src_text.submit(
        fn=translate_wrapper, inputs=[src_text, src_lang, tgt_lang], outputs=[tgt_text]
    )

    clear_btn.click(fn=lambda: ("", ""), outputs=[src_text, tgt_text])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=custom_css,
    )
