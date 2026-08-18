# Confidence tie-break for SSFL voting

## Context

Vote-tie analysis (200 rounds × 3 scenarios, recorded `votes_per_class` matrices) showed ties in
every round; g.tcp+g.udp is the most-tied pair (93,055 sample-rounds), and in scenario 2 the
*entire* flooding class is repeatedly decided by the current tie-break rule "lowest class index
wins" (`ssfl.py:241`, REPRODUCIBILITY #9) — an arbitrary fiat, not a signal.

Goal: break ties using the vote of the client that actually "has the relevant data" —
**without violating protocol assumptions**. The only privacy-safe proxy for "has relevant data"
is the client's own predicted confidence (max softmax): disclosing per-class data ownership
would leak label distributions (excluded); per-sample confidence leaks strictly *less* than the
already-accepted soft-label mode (full 11-dim probability vectors), so it stays inside the
verified privacy boundary.

**Decision (user-confirmed): tie-break only.** Majority vote stays paper-faithful; only ties are
resolved by highest summed client confidence, falling back to lowest index on exact float tie.
Off by default — canonical behavior unchanged.

Honesty note carried into docs: this does NOT recover the collapsed tcp/udp pair (clients cannot
distinguish the two points, so their confidences are near-identical); it replaces an arbitrary
decision with a signal-driven one and removes the systematic index bias.

## Changes

### 1. `src/ssfl/config.py`
- New enum after `LabelRepresentation` (~line 91):
  ```python
  class TieBreak(str, Enum):
      index = "index"
      confidence = "confidence"
  ```
- New field in the SSFL block (lines 195–202): `ssfl_tie_break: TieBreak = TieBreak.index`.
- Extend `_check_ssfl_combinations` (lines 224–247): `confidence` tie-break requires
  `label_representation == hard` and `voting_mode == enabled` (soft mode has no vote ties).

### 2. `src/ssfl/protocols/ssfl.py`
- `aggregate_votes` (lines 194–252): new kwarg `tie_break: TieBreak = TieBreak.index`.
  - Accumulate `conf_sums = np.zeros((num_open, num_classes), float64)` alongside `votes`
    when tie_break is confidence, from `result.confidences` (reusing the existing
    `ProposalResult.confidences` field, which the server path currently sets to `None`).
    Accumulate in **sender-sorted order** for bit-identical determinism (same pattern as
    `aggregate_soft`, line 272).
  - Tie resolution (line 241): when `len(winners) > 1` and confidence mode, winner =
    `winners[np.argmax(conf_sums[i][winners])]`; exact float tie falls back to
    `winners.min()`. `np.argmax` returns first max and `winners` is ascending, so the
    fallback is automatic — but assert-test it explicitly.
  - Confidence mode with a proposal missing confidences → reject that proposal into
    `rejected` (contract violation), don't silently degrade.

### 3. `src/ssfl/client_app.py` `_ssfl_train` (payload at lines 271–300)
- When `exp_config.ssfl_tie_break == confidence` (hard mode): add
  `payload["confidences"] = np.where(labels != ABSTAIN, result.confidences, 0.0)
  .round(2).astype(np.float32)` — zero out abstained samples (leak nothing where no vote is
  cast) and round to 2 decimals (same leak-reduction idea as `ssfl_soft_label_round_decimals`).

### 4. `src/ssfl/protocols/payload_limits.py` `validate_ssfl_proposal_arrays` (lines 43–77)
- New kwarg `allow_confidences: bool = False`.
- Default `False` keeps the hard ban at line 51 — wire contract unchanged for canonical runs.
- When `True`: require `confidences` present alongside `pseudo_labels` (hard mode only),
  shape `(num_open,)`, dtype kind `"f"`, finite, range `[0, 1]`.

### 5. `src/ssfl/strategies/ssfl.py`
- `__init__` (lines 29–46): accept `tie_break`.
- `aggregate_train`: pass `allow_confidences=(tie_break == confidence)` to the validator
  (line 100); unpack `confidences` into `ProposalResult` (line 113, currently `None`); pass
  `tie_break` to `aggregate_votes` (line 127).
- Audit npz (lines 134–145): additionally save `conf_sums` when confidence mode is active,
  so future offline re-voting can replay the tie-break.

### 6. `src/ssfl/server_app.py` (~line 163–170)
- Pass `tie_break=exp_config.ssfl_tie_break` into `SSFLStrategy`.

### 7. `src/ssfl/comms.py` `_paper_array_bytes` (lines 53–65)
- `name.endswith("confidences")` → `array.size * 8` (paper's double-per-probability
  convention; the array is a deviation, so it must be visibly costed, not hidden at 0).

### 8. Tests
- `tests/protocol/test_ssfl.py` (match style of `test_aggregate_votes_majority_tie_and_all_abstain`,
  lines 232–253):
  - tie broken toward higher summed confidence (not lowest index);
  - exact-equal confidences fall back to lowest index;
  - non-tied outcomes identical to index mode (majority untouched);
  - bit-identical under proposal-order shuffle (sender-sorted accumulation);
  - proposal without confidences in confidence mode lands in `rejected`.
- `tests/protocol/test_payload_limits.py`: confidences accepted with `allow_confidences=True`
  (shape/range enforced), still rejected by default — existing
  `test_ssfl_rejects_noncanonical_confidence_upload` (line 38) stays green untouched.
- `tests/unit/test_config.py` (or wherever ssfl combination tests live): confidence tie-break
  rejected with soft representation.
- `tests/unit/test_strategies_security.py`: existing drop-test (line 60) covers default mode;
  add one strategy-level test that confidence mode accepts the payload and `num_proposals == 2`.

### 9. Docs
- `REPRODUCIBILITY.md`: new entry (#31): confidence tie-break — off by default, rationale,
  leakage bound (≤ soft-label mode; rounded, zeroed on abstain), and the honesty note that it
  does not fix the tcp/udp collapse.
- `SECURITY.md`: wire-contract table — `confidences` allowed **only** when
  `ssfl_tie_break=confidence`, forbidden otherwise.

### 10. No new YAML needed
Flag reachable per-run via `run_suite` overrides (generated configs carry any
`ExperimentConfig` key), e.g. an experiments-matrix entry with
`overrides: {ssfl_tie_break: confidence}`. Canonical profiles untouched.

## Verification

```bash
cd "/Users/ayberkkarataban/Federated learning with psuedo labeling" && uv run pytest tests/unit tests/protocol tests/privacy -q
```

```bash
cd "/Users/ayberkkarataban/Federated learning with psuedo labeling" && uv run ruff check .
```

Smoke run with the flag on (CPU, 2 rounds) must complete and the audit npz must contain
`conf_sums`:

```bash
cd "/Users/ayberkkarataban/Federated learning with psuedo labeling" && uv run flwr run . --run-config 'profile="smoke" algorithm="ssfl" scenario=1 device="cpu" ssfl-tie-break="confidence"'
```

(If the run-config key is rejected because `pyproject.toml`'s allowlist doesn't carry it, use a
one-entry `run_suite` matrix with the override instead — do NOT add the key to `pyproject.toml`,
per CLAUDE.md #18.)

Offline sanity check: re-run the tie-analysis script logic on one recorded round with synthetic
confidences to confirm the re-voting math matches `aggregate_votes` output.
