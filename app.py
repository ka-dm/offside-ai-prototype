"""Hugging Face Spaces entrypoint."""

from offside_ai.app.gradio_app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
