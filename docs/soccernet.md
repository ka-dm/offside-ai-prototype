# SoccerNet clips (video + audio)

SoccerNet is the broadcast source for OpenSAOT (semiautomatic VAR): one camera, 25 fps, with an audio track, plus timestamps for **Pass** and **Offside**. It does not label the offside line or the margin in metres. Those stay empty in the CSV (`gt_margin_m`) and come from the pipeline only after you set the attacking direction.

Do not commit the videos. The NDA forbids redistributing them. `artifacts/`, `*.mkv`, and `*.mp4` are gitignored.

## Access

1. Create a Hugging Face account.
2. Open [SoccerNet/SoccerNet_raw_HQ](https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ) and accept the SoccerNet NDA on that dataset. 720p video lives on the `videos-720p` branch. Labels (`Labels-v2.json`) are on [SoccerNet/SN-Labels](https://huggingface.co/datasets/SoccerNet/SN-Labels).
3. Install the downloader (not required to cut clips you already have):

The commands below live in the project virtualenv. Activate it first, or the shell reports `command not found`:

```bash
source .venv/bin/activate
pip install -U SoccerNet huggingface_hub
hf auth login
```

`huggingface-cli login` is the older name of the same login.

## One match, not the full set

Five hundred matches do not fit the paper. Download a single game at 720p (the 2 fps feature files are the wrong source: pass error is measured in milliseconds).

```bash
source .venv/bin/activate
offside-soccernet games --split train --limit 5
offside-soccernet download \
  --local-dir artifacts/soccernet \
  --split train \
  --game "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" \
  --ball-labels
```

That writes:

- `<game>/1_720p.mkv` and `2_720p.mkv` (two halves, with audio)
- `<game>/Labels-v2.json` (action spotting, class `Offside`)

`--ball-labels` also downloads `spotting-ball-2024/<split>.zip` from `SoccerNet/SN-BAS-2024` and unzips it. That zip is the **Pass** labels for the whole split, not 500 videos. `Pass` and `High Pass` are kept; other ball actions are ignored. The visual spot is only an approximate `gt_pass_frame`. The audio onset is measured on the clip.

Without the extra package, the same call is:

```python
from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory="artifacts/soccernet")
dl.downloadGame(
    game="england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
    files=["1_720p.mkv", "2_720p.mkv", "Labels-v2.json"],
    spl="train",
)
```

## Cut windows

Two seconds before the spot and three after, audio re-encoded to AAC. Output stays under `artifacts/`.

```bash
offside-soccernet cut \
  --root artifacts/soccernet \
  --out artifacts/soccernet/windows \
  --pre 2 --post 3 \
  --max-events 12
```

`annotations.csv` uses the columns from `offside-eval schema`:

| Column | What SoccerNet provides |
| --- | --- |
| `gt_pass_frame` | Frame of the spot inside the window |
| `gt_decision` | `offside` only when the label is Offside. A Pass leaves this empty |
| `gt_margin_m` | Empty |
| `attacking_direction` | Empty. Set `positive_x` or `negative_x` before trusting the line |

`python scripts/soccernet_windows.py` is the same CLI.

## Contrast audio with the picture

`offside-audio` finds the impact (spectrogram, band-pass 200–4000 Hz). The pipeline then runs ball motion and `fuse_audio_visual` (agree within 0.12 s, or fall back to the ball when the audio confidence is low).

```bash
offside-soccernet contrast \
  --annotations artifacts/soccernet/windows/annotations.csv \
  --out artifacts/soccernet/contrast \
  --detector mock --pose mock --pitch mock
```

On Colab, switch `--detector rfdetr` and `--pose vitpose`. Predictions record `audio_pass_frame`, `pass_source` (`audio`, `visual`, or `fused`), and the fused `pred_pass_frame`. `report.json` is the pass-frame error against the SoccerNet spot. Decision accuracy is computed only on rows whose label was Offside. The margin in metres is still not ground truth.

Fill `attacking_direction` in the CSV when you know which way the attack runs. Until then the mock line is only a wiring check.
