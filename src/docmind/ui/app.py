"""Build and launch the Gradio web interface."""

from __future__ import annotations

import os

import gradio as gr

from docmind.chat.interface import ConversationHandler
from docmind.engine import KnowledgeEngine
from docmind.ingestion.pipeline import IngestionOrchestrator
from docmind.log import logger
from docmind.ui.theme import STYLESHEET

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")


def _build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    engine = KnowledgeEngine()
    engine.start()

    ingestor = IngestionOrchestrator(
        vector_index=engine.vector_index,
        chunk_repo=engine.chunk_repo,
        splitter=engine.splitter,
    )
    handler = ConversationHandler(engine)

    # -- callbacks ---------------------------------------------------

    def _file_listing() -> str:
        docs = ingestor.list_documents()
        if not docs:
            return "No documents in the knowledge base yet."
        return "\n".join(docs)

    def _on_upload(
        files: list[str] | None,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[None, str]:
        if not files:
            return None, _file_listing()
        added, skipped = ingestor.ingest(
            files,
            on_progress=lambda p, d: progress(p, desc=d),
        )
        gr.Info(f"Added: {added} | Skipped: {skipped}")
        return None, _file_listing()

    def _on_clear() -> str:
        ingestor.wipe()
        gr.Info("Knowledge base cleared")
        return _file_listing()

    def _on_chat(msg: str, hist: list):
        yield from handler.stream_reply(msg, hist)

    def _on_clear_chat() -> None:
        handler.reset_session()

    # -- layout ------------------------------------------------------

    with gr.Blocks(title="DocMind") as demo:
        with gr.Tab("Documents", elem_id="docs-panel"):
            gr.Markdown("## Upload Documents")
            gr.Markdown(
                "Drop PDF or Markdown files below. "
                "Duplicates are skipped automatically."
            )
            file_input = gr.File(
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False,
            )
            upload_btn = gr.Button("Upload", variant="primary", size="md")
            gr.Markdown("## Knowledge Base")
            file_list = gr.Textbox(
                value=_file_listing(),
                interactive=False,
                lines=7,
                max_lines=10,
                elem_id="kb-file-list",
                show_label=False,
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="md")
                clear_btn = gr.Button("Clear All", variant="stop", size="md")

            upload_btn.click(
                _on_upload,
                [file_input],
                [file_input, file_list],
                show_progress="corner",
            )
            refresh_btn.click(_file_listing, None, file_list)
            clear_btn.click(_on_clear, None, file_list)

        with gr.Tab("Chat"):
            avatar = os.path.join(_ASSETS, "bot.png")
            avatar_pair = (None, avatar) if os.path.isfile(avatar) else None
            chatbot = gr.Chatbot(
                height=720,
                placeholder=(
                    "<strong>Ask me anything!</strong><br>"
                    "<em>I will search your documents and "
                    "reason through the answer.</em>"
                ),
                show_label=False,
                avatar_images=avatar_pair,
                layout="bubble",
            )
            chatbot.clear(_on_clear_chat)
            gr.ChatInterface(fn=_on_chat, chatbot=chatbot)

    return demo


def launch() -> None:
    """Create and start the Gradio server."""
    logger.info("Building interface …")
    app = _build_interface()
    logger.info("Launching on http://localhost:7860")
    app.launch(css=STYLESHEET)
