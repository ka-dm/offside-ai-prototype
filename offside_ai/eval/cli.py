"""CLI to score annotation CSVs for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from offside_ai.common import ensure_dir

from .metrics import evaluate_annotations, load_annotations_csv
from .schema import ANNOTATION_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offside-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schema", help="Print / write empty annotation CSV header")
    s.add_argument("-o", "--output", type=str, default=None)

    e = sub.add_parser("score", help="Score predictions against annotations")
    e.add_argument("annotations", type=str)
    e.add_argument("predictions", type=str)
    e.add_argument("-o", "--output-dir", type=str, default="artifacts/eval")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "schema":
        line = ",".join(ANNOTATION_COLUMNS)
        if args.output:
            Path(args.output).write_text(line + "\n")
            print(f"wrote {args.output}")
        else:
            print(line)
        return 0

    ann = load_annotations_csv(args.annotations)
    pred = pd.read_csv(args.predictions)
    report = evaluate_annotations(ann, pred)
    out = ensure_dir(args.output_dir)
    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    pd.DataFrame([e.__dict__ for e in report.error_sources]).to_csv(
        out / "error_sources.csv", index=False
    )
    pd.DataFrame(report.per_clip).to_csv(out / "per_clip.csv", index=False)
    print(json.dumps({"pass_frame": report.pass_frame.to_dict(), "decision": report.decision.to_dict()}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
