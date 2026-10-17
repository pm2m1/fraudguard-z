# Phase 6F — 10k/25k train-prefix scalability acceptance (predeclared)

**Engineering stress test only; not a new fraud-detection study.** Prepared
after 2,000-event in-process acceptance passed and after browser scoring,
duplicate and refresh checks. A Flask-process restart in the browser has
NOT been independently confirmed by screenshot; automated restart checks
are separate.

## Immutable inputs and privacy

- Require the original chronological PaySim CSV. Read **only the first N
  rows** with N in [10,25,000], and reject any step > 238. Never open the
  policy/calibration/validation labels or original final holdout.
- Require the existing verified development `features.npy` (21 columns)
  as the read-only numerical reference. Never load `labels.npy`.
  Confirm source metadata: schema, width, development-only status and
  enough rows. No retraining, recalibration or threshold search.
- Use the checksum-verified local fitted scorer, existing v2 experimental
  REVIEW cutoff, fresh temporary SQLite database and in-process authenticated
  Flask test client; no network listener. Use a distinct, exclusively
  created output JSON. No raw account identifiers, per-event scores or
  credentials in that report.
- Password >=16 characters; private pseudonymization key >=32 UTF-8 bytes.
  This must be validated before model load. Do not rotate a live DB's key.
- Do not touch previous Model B, handbook files, policy v1/v2 outputs
  or the successful 300/2,000-event reports.

## Predeclared checks

- For every replayed event, compare all 21 committed features to the
  original float32 read-only development reference, with
  `np.allclose(rtol=1e-6, atol=1e-6)`.
- Compare reported score to the same verified model evaluated on
  reference features. Verify deterministic evidence against reference
  features and the exact frozen model/policy version.
- Check per-prefix capacity `REVIEW <= floor(events/100)` and BLOCK=0.
  Verify audit idempotency and evidence retrieval across restart.
- Checkpoint at 25%, 50%, 75%, 100% of commits: inspect SQLite main/WAL/
  SHM file sizes **as snapshots**, committed counts and REVIEW counts.
  Restart the journal at 50% and again after 100%, measure full
  open+integrity replay duration. Check SQLite `PRAGMA quick_check`
  is `ok`. Verify all data survives and duplicates don't change counts.
- Time each **in-process Flask test_client POST including SQLite commit**
  (p50/p95/p99/mean/max). Record total wall time separately from request
  times and from full restart/rebuild. Report bounded aggregate metrics;
  this is not network, concurrency, production or IBM LinuxONE latency.
- No numerical performance or fraud-recall target is inferred from this
  engineering-only replay. Preserve failures as failures: never publish
  `PASS` until all tests/checkpoints succeeded.
- `--count` defaults to 10,000 and maximum is 25,000. Use a NEW report
  file for a later 25,000-event test; never overwrite earlier reports.

## Limitations and next decisions

In-process read-only reference parity shares the project's feature
semantics; this is not independent implementation verification.
A successful benchmark does not redeem policy v2: it flagged 97/412
frauds (23.54%) and 0/206 CASH_OUT frauds on already-inspected
PaySim policy data. BLOCK stays disabled; no live payment use.
An independent simulator/dataset and fair prospective confirmation,
plus actual network/browser security acceptance, are still outstanding.
