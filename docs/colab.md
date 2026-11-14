# Running on Google Colab (Kevin)

Paper deadline: **13 Nov 2026**. Fast path to try 3–4 plays:

1. Open `notebooks/offside_colab.ipynb` in Colab (**Runtime → GPU** if you want ViTPose / RF-DETR / YOLO).
2. Run the install cell (`requirements-colab.txt` + `pip install -e .`).
3. Upload a **short** clip (a few seconds around the pass).
4. Run the **audio stage first** — inspect the spectrogram and candidate frames.
5. If confidence is low, set `manual_pass_frame` and continue.
6. Run the full pipeline with `detector=rfdetr` or `yolo`, `pose=vitpose` when weights download succeeds; otherwise keep `mock` for UI wiring checks.

## What this VM verified vs what you must verify on Colab

| Piece | Verified in CI / this environment | You should verify on Colab |
|-------|-----------------------------------|----------------------------|
| Audio onset detection (synthetic WAV + muxed tiny MP4) | Yes (pytest) | Real broadcast clips |
| Offside geometry + annotation | Yes (pytest) | Homography on real pitch views |
| Eval metrics + synthetic CSV | Yes (pytest) | Your annotated play set |
| RF-DETR / YOLO weights download | **No** (not pulled here) | Install + first detect |
| ViTPose weights | **No** | Install transformers + download |
| SAM 3D Body | Scaffold only | Only if license accepted |
| Gradio app import | Installable; mock path | Optional Space deploy |

## Tips

- Set `--av-offset` if audio and video are skewed (common in downloaded clips).
- Prefer RF-DETR (Apache-2.0) over YOLO (AGPL-3.0) for the paper’s reproducibility appendix.
- Never commit copyrighted footage to GitHub.
