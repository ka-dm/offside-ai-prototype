# Evaluation fixtures

This folder holds **non-copyrighted** tiny samples for CI and local smoke tests.

- Generated WAVs / CSVs are created on the fly by `offside_ai.eval.synthetic.build_synthetic_eval_dir`.
- Do **not** commit broadcast football footage or other copyrighted media.
- For the paper dataset, keep clips offline and list them in a private `annotations.csv` using the schema from `offside-eval schema`.

Example:

```bash
python -c "from offside_ai.eval.synthetic import build_synthetic_eval_dir; build_synthetic_eval_dir('artifacts/syn_eval')"
offside-eval score artifacts/syn_eval/annotations.csv artifacts/syn_eval/predictions.csv -o artifacts/syn_eval/out
```
