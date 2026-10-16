# FDH replication incident 01 — first-run numerical underflow

The original one-time locked replication (branch research/fdh-seeded-generator, locked commit d233ce7) was started locally with `python -m src.fdh_replication_runner --run`.

The first announced run `seed_104729__default` failed before a per-run output was saved, with `RuntimeWarning: overflow encountered in scalar subtract` followed by `RuntimeError: Paired detection accounting inconsistent`. The problem arose because NumPy uint8 label sums were subtracted before conversion to Python int; negative differences can underflow. **This is an evaluator bug, not a model-quality verdict.** The error indicates the history model may have detected fewer frauds in this first case, but no valid detection delta was saved.

Original source code and lock remain frozen on the original branch. The local `reports/fdh_replication_started.json` and `reports/fdh_replication_runs/` MUST be preserved; no deletion, overwrite or silent rerun. Run files may be absent because the first run failed prior to write.

Remediation branch `research/fdh-replication-incident-01` changes only the paired count subtraction to `int(final_y[hist].sum()) - int(final_y[static].sum())`. It adds regression tests, then requires a new separately versioned **amended protocol and hash lock** if a fresh attempt is explicitly approved. The original pre-generation claims no longer apply to the amended rerun of the seed observed in the failed attempt. State that the first seed/regime was touched and yielded an error indicating a negative difference; any future result from the same nine configs is a transparent exploratory amended replication, not untouched confirmatory evidence.

Avoid claiming success based on 115 prior tests; they missed this underflow case. Do not generate data or invoke `--run` on this branch yet. Review other unsigned arithmetic and ensure new tests pass.
