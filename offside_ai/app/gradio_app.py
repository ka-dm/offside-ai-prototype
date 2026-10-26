"""Gradio app: upload clip → pass frame + spectrogram → offside verdict."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from offside_ai.pipeline.run import PipelineConfig, run_pipeline


def _run(
    video,
    attacking_team_id,
    attacking_direction,
    av_offset,
    manual_pass_frame,
    detector,
    pose,
    pitch,
):
    if video is None:
        raise gr.Error("Upload a short video clip first.")
    video_path = video if isinstance(video, str) else getattr(video, "name", None)
    if not video_path:
        raise gr.Error("Could not resolve uploaded video path.")

    manual_frame = None
    if manual_pass_frame is not None and str(manual_pass_frame).strip() != "":
        try:
            manual_frame = int(manual_pass_frame)
        except ValueError as e:
            raise gr.Error("Manual pass frame must be an integer") from e

    work = Path(tempfile.mkdtemp(prefix="offside_app_"))
    cfg = PipelineConfig(
        detector_backend=detector,
        pose_backend=pose,
        pitch_backend=pitch,
        attacking_team_id=int(attacking_team_id),
        attacking_direction=str(attacking_direction),
        av_offset_s=float(av_offset or 0.0),
        manual_pass_frame=manual_frame,
        work_dir=str(work),
    )
    try:
        result = run_pipeline(video_path, cfg)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(str(e)) from e

    verdict = result.decision.verdict
    detail = (
        f"Pass: frame {result.pass_event.frame_index} "
        f"({result.pass_event.time_s:.3f}s, source={result.pass_event.source}, "
        f"conf={result.pass_event.confidence:.2f})\n"
        f"Offside line x={result.decision.offside_line_x:.2f} m\n"
        f"Offside attackers: {result.decision.offside_attacker_ids}\n"
        f"Margins (m): {result.decision.margins_m}\n"
        f"Backends: det={result.meta.get('detector')} "
        f"pose={result.meta.get('pose')} pitch={result.meta.get('pitch')}"
    )
    return (
        verdict,
        detail,
        result.spectrogram_path,
        result.annotated_path,
        result.pass_event.frame_index,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Offside AI Prototype") as demo:
        gr.Markdown(
            """
            # Offside AI — student research prototype
            Upload a **short** single-camera broadcast clip. The system estimates the
            pass/kick instant from **audio** (with optional visual fallback), then
            produces an offside verdict. Defaults use lightweight **mock** detectors
            so the UI runs without downloading large weights; switch to RF-DETR / YOLO
            / ViTPose on Colab or a GPU machine.
            """
        )
        with gr.Row():
            video = gr.Video(label="Clip")
            with gr.Column():
                attacking_team = gr.Number(value=0, precision=0, label="Attacking team id")
                direction = gr.Dropdown(
                    ["positive_x", "negative_x"],
                    value="positive_x",
                    label="Attacking direction",
                )
                av_offset = gr.Number(value=0.0, label="A/V offset (s)")
                manual_frame = gr.Textbox(
                    value="",
                    label="Manual pass frame (optional override)",
                    placeholder="e.g. 42",
                )
                detector = gr.Dropdown(
                    ["mock", "rfdetr", "yolo"], value="mock", label="Detector"
                )
                pose = gr.Dropdown(
                    ["mock", "vitpose"], value="mock", label="Pose backend"
                )
                pitch = gr.Dropdown(
                    ["mock", "hf"], value="mock", label="Pitch backend"
                )
                run_btn = gr.Button("Run", variant="primary")

        verdict = gr.Textbox(label="Verdict")
        detail = gr.Textbox(label="Details", lines=8)
        with gr.Row():
            spectrogram = gr.Image(label="Spectrogram + onsets")
            annotated = gr.Image(label="Annotated pass frame")
        pass_frame = gr.Number(label="Detected pass frame", precision=0)

        run_btn.click(
            _run,
            inputs=[
                video,
                attacking_team,
                direction,
                av_offset,
                manual_frame,
                detector,
                pose,
                pitch,
            ],
            outputs=[verdict, detail, spectrogram, annotated, pass_frame],
        )
    return demo


def main() -> None:
    demo = build_app()
    demo.launch()


if __name__ == "__main__":
    main()
