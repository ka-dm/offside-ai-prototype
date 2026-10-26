# Offside AI Prototype

Semi-automated **football offside detector** for a university paper (due **13 Nov 2026**). Input: a short single-camera broadcast clip. Output: estimated **pass/kick instant** + **offside verdict** (line, margins, annotated frame). The paper compares this workflow to traditional VAR.

**Novelty:** pass-time from **audio** (spectrogram + band-pass onset detection), with visual ball-motion cross-check and manual fallback when crowd noise makes audio unreliable.

## Package layout

```
offside_ai/
  audio/       # kick/pass from audio (+ visual fuse)
  detection/   # RF-DETR / YOLO / mock + jersey clustering
  pitch/       # keypoints + homography (m)
  pose/        # ViTPose / mock; SAM 3D Body flagged off
  offside/     # geometry + annotation
  pipeline/    # end-to-end wiring
  app/         # Gradio UI
  eval/        # CSV schema + metrics for the paper
notebooks/     # Colab quickstart
docs/          # architecture (mermaid) + licenses
tests/         # pytest (audio, geometry, eval)
```

Each stage is independently testable via a CLI (`offside-audio`, `offside-detect`, …).

## Quick start (laptop)

```bash
# system: ffmpeg
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

pytest
```

Audio-only on a clip:

```bash
offside-audio path/to/clip.mp4 -o artifacts/audio
```

Full pipeline with lightweight mocks (no model weights):

```bash
offside-pipeline path/to/clip.mp4 --detector mock --pose mock --pitch mock -o artifacts/run
```

Gradio:

```bash
offside-app
# or: python app.py
```

## Google Colab (Kevin)

1. Open [`notebooks/offside_colab.ipynb`](notebooks/offside_colab.ipynb) (GPU runtime recommended for real detectors/pose).
2. Install (`requirements-colab.txt` + editable package); ensure **ffmpeg** is present.
3. Upload a **few-second** clip around the pass.
4. Run the **audio stage first** — check spectrogram peaks and confidence.
5. If `confidence < ~0.45`, set `MANUAL_PASS_FRAME` and continue.
6. Switch `--detector rfdetr` (or `yolo`) and `--pose vitpose` once weights download on Colab.

More detail: [`docs/colab.md`](docs/colab.md).

## Evaluation (paper)

CSV schema: `offside-eval schema`. Example columns in `eval/fixtures/annotations.example.csv`.

```bash
python -c "from offside_ai.eval.synthetic import build_synthetic_eval_dir; build_synthetic_eval_dir('artifacts/syn_eval')"
offside-eval score artifacts/syn_eval/annotations.csv artifacts/syn_eval/predictions.csv -o artifacts/syn_eval/out
```

Reports: pass-frame error (frames + ms), decision accuracy / precision / recall, margin error, error-source table. **Do not commit copyrighted footage.**

## What was verified vs not

| Component | Status in this repo / CI |
|-----------|---------------------------|
| Audio onset detection (synthetic WAV, muxed tiny MP4) | **Tested** (`pytest`) |
| Visual ball-motion fuse helpers | **Tested** |
| Offside geometry + annotation | **Tested** |
| Eval metrics + synthetic fixtures | **Tested** |
| Pitch homography math / manual calibration | **Tested** |
| Detection team clustering (mock boxes) | **Tested** |
| Pipeline wiring with **mock** backends | Implemented; run locally on any short clip with audio |
| RF-DETR / YOLO / ViTPose **weight downloads** | **Not verified** in the cloud VM (no large weight pull) |
| SAM 3D Body | Scaffold only (`--enable-sam3d`) |
| Real broadcast accuracy | **Not measured** here — use `eval/` on your annotated set |

Be honest in the paper: single-camera homography ≠ broadcast VAR multi-view triangulation; audio helps pass timing but fails under loud crowds.

## Licenses

- **Our code:** MIT (`LICENSE`)
- **RF-DETR:** Apache-2.0 (default detector)
- **ViTPose:** Apache-2.0 (verify the HF card you pin)
- **Ultralytics YOLO:** AGPL-3.0 (optional)
- **SAM 3D Body:** gated SAM License (optional, off by default)

See [`docs/LICENSES.md`](docs/LICENSES.md) and [`docs/architecture.md`](docs/architecture.md).

## Hugging Face Space

Entrypoint: `app.py`. Space README template: `README_SPACE.md` (YAML frontmatter + Gradio SDK). Defaults to **mock** backends so the Space boots without large downloads.

## Español (breve)

Prototipo de investigación: detecta el **momento del pase** sobre todo con **audio** (espectrograma + onsets) y calcula fuera de juego con homografía y puntos corporales legales (sin brazos). En Colab: abre el notebook, sube un clip corto, revisa primero la etapa de audio; si la confianza es baja, elige el frame a mano. No subas vídeo con copyright al repositorio.

## Citation / course use

Internal student prototype for coursework — not a certified officiating tool.
