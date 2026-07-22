# Security and Privacy

Threat model, verified privacy properties, and the concrete defenses in this codebase, per
`SSFL_IMPLEMENTATION_PLAN.md` Milestones M8 (Privacy and Security) and M9 (Reliability and
Observability). Written in the same spirit as `REPRODUCIBILITY.md`: state what is actually true of
the code today, not what would be nice to claim.

## Threat model

| Actor | Capability assumed | What this build defends against |
|---|---|---|
| Honest-but-curious server | Observes every message it legitimately receives; does not deviate from protocol | Never receives private features, labels, gradients, or model parameters from any of the four algorithms (see "Privacy boundary" below) — there is nothing sensitive in the messages to be curious about. |
| Curious client | Observes its own messages and any broadcast payload; does not deviate from protocol | Broadcasts (`global_labels`/`valid_mask`, FD leave-self-out targets, DS-FL sharpened targets) never contain another client's raw predictions or identity-linked data. |
| Network observer (passive, on the wire between clients and server) | Sees message sizes/timing/plaintext content in the simulation backend; sees only TLS-wrapped bytes when the `tls` deployment profile is used | **Not defended against in the simulation backend** — Flower's in-process/Ray simulation transport is unencrypted by construction. The real (non-simulation) deployment scaffold in `deployment/` (see "Deployment gap" below) supports both an `insecure` dev profile and a `tls` profile (self-signed dev CA via `deployment/certs/generate_dev_certs.sh`, `--ssl-certfile`/`--ssl-keyfile`/`--ssl-ca-certfile` on the SuperLink, `--root-certificates` on each SuperNode); the simulation backend itself is unaffected either way. |
| Compromised client (sends arbitrary bytes, not just off-protocol values) | Full control over its own client process's messages | `validate_ssfl_proposal_arrays`/`validate_fd_arrays`/`validate_dsfl_arrays` (`src/ssfl/protocols/payload_limits.py`) reject wrong-shape, wrong-dtype, non-finite, or out-of-probability-range payloads before they reach aggregation. Sender-authorization (below) rejects a compromised node impersonating a sender ID that isn't in the current round's sampled cohort. |
| Malicious pseudo-label contributor | A sampled, authorized client that submits well-formed but adversarial votes/labels | Majority voting (`aggregate_votes`) bounds a single client's influence to one vote per open example; `aggregate_soft`'s mean similarly dilutes one client's contribution across the cohort. No defense against a *majority* of sampled clients colluding — matches the plan's "optional extension profiles" (robust aggregation, client reputation, minimum vote quorum) being explicitly out of scope for paper mode. |
| Model/membership inference | Observes broadcast labels/targets and correlates with known query points | **Not defended against, and not fully preventable by this protocol family** — see "What SSFL does not claim" below. |
| Replay / stale-message attacks | Resubmits a previously valid message, or a message for the wrong round/phase | `message_id` dedup (`protocols/message.py::validate_envelope`) rejects a byte-identical resubmission within `SSFLStrategy.aggregate_train`. **Known limitation**, not fixed this pass: see below. |

## Privacy boundary (verified)

Per-protocol, confirmed by reading `client_app.py` and each `strategies/*.py`:

- **SSFL**: client → server carries `confidences`, `pseudo_labels` *or* `soft_probs`, and scalar
  losses only (`protocols/ssfl.py::ProposalResult`). Server → client carries `global_labels` +
  `valid_mask` only. No model parameters, gradients, or private features cross the wire in either
  direction — enforced structurally (the message schema has no field for them), and covered by
  `tests/protocol/test_ssfl.py`.
- **FD**: client → server carries `class_probs` (per-class mean softmax, never a raw per-example
  prediction) + `class_present`. Never a state_dict.
- **DS-FL**: client → server carries `probs` over the shared *open* (unlabeled, non-private) set
  only — never the private partition.
- **FL** is the one algorithm that legitimately transmits model parameters (stock `FedAvg`) — this
  is expected and documented, not a gap; SSFL/FD/DS-FL exist specifically to avoid this.

### What SSFL does not claim

Per the plan's explicit M8 instruction ("hard labels can still leak model behavior; documentation
must avoid claiming complete privacy"): a `global_labels`/`soft_probs` broadcast is a function of
every contributing client's model, so a sufficiently well-resourced adversary who can query the
open set and observe many rounds of broadcasts could still mount a model-extraction or, in
principle, a membership-inference attack against the federation as a whole. This build gives clients
and the server no access to each other's raw private data/parameters; it does not give
information-theoretic privacy against inference from aggregated, repeated outputs. Differential
privacy is listed in the plan as an optional, disabled-by-default extension for exactly this reason.

## Payload-level defenses (implemented this pass)

1. **Sender authorization.** `SSFLStrategy`/`FDStrategy`/`DSFLStrategy`'s `aggregate_train` and
   `aggregate_evaluate` now all drop any reply whose `msg.metadata.src_node_id` is not in the
   current round's sampled `self._current_node_ids`. Before this, **FD and DS-FL had zero
   sender-identity checking** — any node ID Flower delivered a reply from was aggregated
   unconditionally. SSFL already had this check via `validate_envelope`'s `valid_senders` set; FD/DS-FL
   now match it directly (a lighter-weight check, since they don't use the full `Envelope` machinery).
2. **Shape/dtype/range validation.** `protocols/payload_limits.py` validates every array a client
   returns before it is consumed: correct shape, float vs. int dtype, finite (no NaN/Inf), and
   value range (probabilities in `[0,1]`, rows sum to ~1, pseudo-labels in `[-1, num_classes)`).
   Before this, a malformed reply (wrong shape, a label index like `99`, a NaN confidence) would
   either crash `aggregate_train` for the whole round or silently corrupt the vote/mean it feeds
   into — this is the concrete mechanism behind the "malicious pseudo-label contributor" and
   "payload-size and shape limits" threat-model entries.
3. Both checks reject-and-continue (log a rejection, keep processing the rest of the round) rather
   than raising — one bad client can no longer take down aggregation for the whole cohort.
4. Tests: `tests/protocol/test_payload_limits.py` (pure validators, 14 cases) and
   `tests/unit/test_strategies_security.py` (proves the real `flwr.common.Message` path actually
   drops an unauthorized sender / malformed payload for all three strategies, 6 cases).

### Known limitation: envelope validation is not yet end-to-end

`SSFLStrategy.aggregate_train` builds the `Envelope` it validates (`algorithm`, `scenario`, `round`,
`phase`, `dataset_manifest_hash`) entirely from **server-side state**, not from anything the client
actually put in its message — `client_app.py`'s replies carry no such fields today. This means
`validate_envelope`'s algorithm/scenario/round/phase/hash checks can structurally never fail as
currently wired; only `sender_id`-based rejection and `message_id` dedup provide real protection
from this path. In practice this is lower-risk than it sounds: Flower's own SuperLink matches each
round's replies to that round's own outstanding requests via `reply_to`/`message_id` before
`aggregate_train` ever sees them, so a genuinely stale reply from a prior round has nowhere to be
delivered to in the first place — the gap is real for a compromised SuperNode that tampers with
reply *content* inside an otherwise legitimately-routed message (which sender-authorization and
payload-limits above do cover), not for cross-round replay via the transport layer. Making the
envelope check itself non-tautological would require every protocol's client reply to echo back
`algorithm`/`scenario`/`round`/`phase`/`dataset_manifest_hash` (likely as extra `MetricRecord`
fields) and the strategies to build `Envelope` from those instead — not done in this pass; flagged
here rather than silently left implying a stronger guarantee than exists.

## Deployment gap

`deployment/generate_launch_configs.py` emits real (non-simulation) SuperLink + SuperNode launch
scripts from a scenario manifest, for both an `insecure` dev profile and a `tls` profile
(`deployment/certs/generate_dev_certs.sh` generates a self-signed dev CA + SuperLink leaf cert).
Verified live: SuperLink starts and 3 SuperNodes register against it (`Fleet.ActivateNode`) for
both profiles, with clean shutdown — see `tests/deployment/test_deployment_smoke.py`
(`@pytest.mark.slow`; run with `pytest -m slow tests/deployment`).

**Scope, not a gap:** `ExperimentConfig.num_clients()` is a fixed mapping (scenario 1 = 27, 2/3 =
89) and the custom strategies (`SSFLStrategy`/`FDStrategy`/`DSFLStrategy`) require an exact
client-count match with no partial-participation support, so a *full federated round* with only 3
clients is structurally impossible — the smoke test verifies process-level wiring (registration,
connectivity, clean shutdown) using a 3-of-27 subset instead. A full-count run
(`--num-clients` omitted, i.e. all 27/89) uses the same generator and scripts; it just needs that
many real `flower-supernode` processes, which is a hardware/orchestration concern, not a missing
feature here.

**Not yet built:** node authentication (`--enable-supernode-auth`/`--auth-list-public-keys` exist
on the installed Flower CLI but aren't wired into the generator), certificate rotation (the dev
certs are `--days 825`, static, no renewal), secret injection beyond passing cert paths as CLI
flags, container images, and dependency/container vulnerability scanning. All *training* work in
this repo (the 200-round paper-profile runs, all ablations/studies) still runs through Flower's
local-simulation backend only — the real deployment scaffold has been verified for process wiring,
not used to run an actual multi-round experiment.

There is no "manifest-hash check at registration" to test: Flower's SuperNode registration
handshake (`Fleet.ActivateNode`) is transport-level and does not carry the app's
`ConfigRecord`/`ArrayRecord` payloads. `dataset_manifest_hash` validation
(`protocols/message.py::validate_envelope`) only fires on the first real train/evaluate message
exchange, which needs the full client count for that scenario — out of scope for a 3-client smoke
test, in scope for (and already covered by) the simulation-backend integration tests in
`tests/integration/test_simulation_smoke.py`.

No credentials are committed to this repository (verified: no `.env`, no key/cert files, no secrets
in `configs/*.yaml`).

## Logging

`src/ssfl/logging_utils.py` provides structured JSON-lines logging (`configure_logging`/`bind`/
`log_event`) with a forbidden-field guard (`ForbiddenLogFieldError`) that refuses to log any field
whose key matches a private-data/model-tensor/secret pattern (`_FORBIDDEN_KEY_SUBSTRINGS`). Until
this pass this module existed but **was never called from `server_app.py` or `client_app.py`** —
`run_context.py`'s `events.jsonl` was a documented-but-never-written file. Now wired:

- `server_app.py::main()` opens `<run_dir>/events.jsonl` and binds a logger with
  `run_id`/`algorithm`/`scenario`, emitting `run_start` (with `dataset_hash`) and `run_end` (with
  `final_round`) events.
- `comms.py::CommsTrackingStrategy` — the single point every algorithm's `aggregate_train`/
  `aggregate_evaluate` already passes through — emits one `aggregate` event per round/phase with
  `round`, `phase`, `num_replies`, and every numeric scalar the inner strategy's metrics returned
  (e.g. SSFL's `valid_rate`/`tie_count`/`all_abstain_count`/`rejected_count`). It never touches
  `arrays_out`, so model weights/pseudo-labels/soft-probs cannot end up in the log by construction —
  the forbidden-field guard is defense-in-depth on top of that structural exclusion, not the only
  thing standing between this code and logging something sensitive.
- Client-side (`client_app.py`) logging is **not yet wired** — deferred, since the server-side
  aggregation log already captures round/phase/rejection observability without needing every
  client process (there can be up to 89) to also log independently.
- Verified end-to-end against a real Flower CPU simulation, not just the wiring compiling: a live
  SSFL smoke run initially produced an `events.jsonl` where every per-call field (`round`, `phase`,
  `num_replies`, `valid_rate`, `tie_count`, ...) was silently missing — a bug in `log_event`'s
  interaction with `bind()`'s `LoggerAdapter` (see `REPRODUCIBILITY.md` #26), now fixed and covered
  by `tests/unit/test_logging_utils.py` and a `logging_utils.py` self-check. Re-running the same
  smoke confirmed `events.jsonl` now carries the full field set.

## Metrics

Covered today: test-set accuracy/precision/recall/F1 (`metrics.py::MetricsLedger`, evaluate-phase
only), per-round communication bytes (`comms.py::CommsLedger`, logical + serialized), valid
pseudo-label rate / tie count / all-abstain count / accepted-proposal count / **rejected reply
count** (new this pass) for SSFL, and an equivalent `rejected_count` for FD/DS-FL — all flow into
the `aggregate` log events above.

Not covered (flagged, not built this pass — would need real multi-process deployment or dedicated
instrumentation neither of which exist yet): active/ready/failed client counts, round/phase wall-clock
latency as a first-class metric (only whole-run wall-clock is captured today), per-class vote
distribution as a *persisted* artifact (`AggregationResult.votes_per_class` is computed in-memory
every round but never written anywhere), and memory/compute utilization.

## Recovery

**Corrected during this pass** — an earlier draft of this section claimed mid-run checkpoint/resume
"exists and [was] verified by earlier milestones." That was wrong; found by grepping for actual
call sites, the same way the logging-was-dead-code gap (above) was found. What actually exists:

- `RunContext.checkpoints_dir`, `RunContext.last_completed_round()` (scans `checkpoints/round_*.pt`),
  and `RunContext.resume()` (re-validates `resolved_config.yaml` against the current
  `ExperimentConfig` schema) are all implemented in `run_context.py`, and `ExperimentConfig` has
  `checkpoint_interval`/`checkpoint_rounds`/`resume_from` fields.
- **Nothing calls any of them from the real training path.** `server_app.py::main()` always calls
  `RunContext.create(...)`, never `RunContext.resume(...)`; nothing ever writes a
  `checkpoints/round_<N>.pt` file; `resume_from` is read nowhere. `last_completed_round()` would
  always return `0` in practice. This means a single `flwr run` interrupted mid-training (e.g. at
  round 87/200) has no way to resume from round 88 — it can only be restarted from round 1.
- The one form of resume that *does* work: `run_suite.py --resume` skips a matrix entry whose
  `summary.json` already exists — i.e. it resumes an interrupted *experiment matrix* (skip
  already-fully-completed runs), not an interrupted *single run*. Do not conflate the two when
  reading `REPRODUCIBILITY.md` or `run_suite.py`'s docstring.
- Building real mid-run checkpoint/resume (persisting the strategy's `ArrayRecord` state per round,
  restoring it into `Strategy.start()`, handling Ray/simulation-process restart) is out of scope for
  this pass — flagged here rather than left as a silently false "done" claim. See
  `REPRODUCIBILITY.md` #27.

Not verified this pass (separate from the above, flagged as open questions): configurable
retry/timeout limits on a hung client; explicit behavior when fewer than the expected number of
clients respond in a round (`Strategy.start()`'s built-in `min_*_nodes` semantics apply to FL's
stock `FedAvg`; the three custom strategies have no equivalent minimum-participation check — an
SSFL round with only 1 of 27 clients responding would proceed with whatever plurality that gives);
whether cancelling a run mid-round preserves previously-completed rounds' results (`metrics.parquet`/
`communication.parquet` are written once at the very end of `main()`, not incrementally — so killing
a run mid-training currently loses ALL of that run's metrics/comms data, not just the incomplete
round; only the identity files written by `RunContext.create()` up front would survive).

## Optional extension profiles (out of scope, per plan)

Differential privacy, minimum vote quorum, robust aggregation, client reputation, secure
aggregation — none implemented. The plan states these "remain disabled in paper mode," which is
the only mode this build targets.
