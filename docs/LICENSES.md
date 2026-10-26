# Licenses of models and datasets

Our application code in this repository is **MIT** (see `/LICENSE`). Third-party models and datasets keep their own terms. Review each model card before redistribution (especially Hugging Face Spaces).

| Component | Role | License | Notes |
|-----------|------|---------|-------|
| **RF-DETR** (Nano → Large) | Default detector | **Apache-2.0** | Preferred default for open redistribution |
| **ViTPose** (HF / community ports) | Body keypoints | **Apache-2.0** (typical for official / usyd-community ports — verify the card you pin) | Arms excluded in our legal-point logic |
| **Ultralytics YOLO** | Optional detector | **AGPL-3.0** | Copyleft; using it may affect how you distribute a combined service. Prefer RF-DETR for MIT-friendly stacks |
| **SAM 3D Body** | Optional 3D body | **Gated SAM License** | Off by default (`--enable-sam3d`). Accept license on Hugging Face before download; heavy |
| **librosa / scipy / numpy / OpenCV** | Audio & geometry | BSD / Apache-style (see each project) | Core audio stage |
| **Gradio** | UI | Apache-2.0 | App / Spaces |
| **roboflow/sports** (reference) | Pitch / sports helpers | Check upstream repo | Inspiration for pitch calibration; not vendored by default |
| **Synthetic eval fixtures** | CI / smoke tests | Generated here (MIT) | **No copyrighted broadcast footage** is committed |

## Datasets for the paper

Annotated plays for the university paper should be stored **outside** this public repo if they contain copyrighted broadcast video. Use `eval/fixtures/annotations.example.csv` as the schema and keep `credits` / `source` columns filled for your methods section.

## Hugging Face Spaces

If you deploy with YOLO weights, AGPL obligations may apply to the Space. The default Space configuration uses **mock** detection/pose/pitch backends so the demo boots without pulling AGPL or gated weights.
