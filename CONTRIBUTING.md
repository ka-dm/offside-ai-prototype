# Contributing to OpenSAOT

OpenSAOT is a research prototype of a semiautomatic offside check for a short, single-camera broadcast clip. It estimates the pass or kick instant, then the offside line, the margin, and an annotated frame.

The code in this repository is MIT. Models and datasets you plug in keep their own licenses. Read [docs/LICENSES.md](docs/LICENSES.md) before adding a weight file or a new backend.

## What helps

- A bug fix or a test that fails on `main` and passes after the change.
- A clearer fallback when audio confidence is low.
- Geometry, homography, or evaluation metrics, with a unit test.
- Documentation that matches the commands in the README.

Do not open a pull request that only reformats the tree or swaps the default detector for one with a stricter license.

## Setup

You need Python 3.10 or newer and FFmpeg.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```

CI runs `pytest` on Python 3.10 and 3.12. It installs the core package only. It does not download RF-DETR, YOLO, or ViTPose weights. New tests must pass with the `mock` backends.

Useful commands while you work:

```bash
offside-audio path/to/clip.mp4 -o artifacts/audio
offside-pipeline path/to/clip.mp4 --detector mock --pose mock --pitch mock -o artifacts/run
offside-eval schema
pytest tests/audio tests/offside tests/eval
```

Each stage has its own CLI: `offside-audio`, `offside-detect`, `offside-pitch`, `offside-pose`, `offside-decide`, `offside-eval`, `offside-pipeline`, `offside-soccernet`, `offside-app`, and `offside-studio`.

## How to change a stage

Keep the stage importable and testable without the rest of the pipeline.

| Area | Package | What a change should still do |
| --- | --- | --- |
| Pass or kick instant | `offside_ai/audio/` | Return a frame and a confidence. Low confidence must still allow a manual frame. |
| Players and ball | `offside_ai/detection/` | Support `mock`. Do not make a heavyweight model the only path. |
| Pitch in metres | `offside_ai/pitch/` | Keep the homography testable without a checkpoint. |
| Body points | `offside_ai/pose/` | Exclude the arms from the legal offside point. Leave SAM 3D Body behind `--enable-sam3d`. |
| Verdict | `offside_ai/offside/` | Stay a pure geometry step. |
| Paper numbers | `offside_ai/eval/` | Extend `annotations.example.csv` if you add a column, and cover it in `tests/eval`. |

If you add a dependency, put the lightweight one in `requirements.txt` and the GPU or model one in `requirements-colab.txt` or the matching extra in `pyproject.toml`.

## Media, weights, and secrets

- Do not commit broadcast footage, SoccerNet videos, or other copyrighted match video. Synthetic clips and the CSV schema are the fixtures CI is allowed to see.
- Do not commit model weights, Hugging Face tokens, or API keys.
- SoccerNet clips stay outside the repo. The NDA workflow is described in [docs/soccernet.md](docs/soccernet.md).
- Annotated plays for the paper stay outside this public tree when they contain broadcast video. Keep `credits` and `source` filled in the CSV.

## Licenses when you add a model

- Prefer **RF-DETR** (Apache-2.0) as the detector.
- **Ultralytics YOLO** is AGPL-3.0. A pull request that turns it on by default needs a note in `docs/LICENSES.md` and in the Space README, because it can change how a combined service may be distributed.
- **SAM 3D Body** is a gated license. Leave it off unless the user passes `--enable-sam3d`.
- Pin the model card you actually tested. Do not claim an Apache-2.0 port without naming the repository.

## Pull requests

1. Branch from `main`.
2. Describe the behaviour you changed, not only the files.
3. Add or update a test under `tests/` for the stage you touched.
4. Run `pytest` locally.
5. If the change affects the Colab path or the Gradio app, say what you ran and what you could not run (for example, no GPU, no weights).
6. Do not report a broadcast accuracy number unless it comes from `offside-eval` on an annotated set. Single-camera homography is not multi-camera VAR.

A short pull request is easier to review than a branch that retunes every stage at once.

## Reporting a problem

Include the command, the Python version, whether you used `mock` or a real weight, and the clip length. If audio confidence collapsed, say whether the clip has crowd noise. Do not attach the match video to the issue if you do not have the right to share it. A few seconds of a synthetic or self-recorded clip is enough.
