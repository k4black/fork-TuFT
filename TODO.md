# Roadmap & Fork TODO

Based on `../llmqa-team-junk/docs/research/2026-09-11-tuft-fork-review.md` and repository inspection.
Target branches:
- `main`: Active development & PR merge target.
- `upstream`: Tracking `agentscope-ai/TuFT:main`.

All implementations must follow `/ponytail` (minimal code, reuse existing patterns, no bloat) and use isolated git worktrees. Features that are not small fixes require plans and `/grill-me` alignment before implementation.

---

## Phase 0: Quick Wins, Versions & Small Fixes
- [x] **Dependency & vLLM Alignment**
  - Upgraded coordinated `torch==2.13.0` and `vllm==0.27.0` in `pyproject.toml`.
  - Pinned `KNOWN_UNTAGGED_BACKENDS` for vLLM 0.27.0 in `scripts/verify_runtime_versions.py` and updated runtime tests.
  - Adapted `create_server_socket(addr, reuse_port=False)` with backwards-compatible fallback in `src/tuft/backends/vllm_api_server.py`.
  - CI Checks (3.11, 3.12, 3.13) passing green. (Merged in #8)
- [x] **CI Lint & Formatting Fixes**
  - Cleaned all import formatting, line-length warnings, and Pyright type issues.
  - Added dependabot rules to ignore isolated single-package bumps of `torch` and `vllm`. (Merged in #7)
- [x] **C6: LoRA Alpha Configuration** (`src/tuft/config.py`, `src/tuft/backends/fsdp_training_backend.py`, `src/tuft/checkpoints.py`)
  - Remove strict integer `>= 1` limitation on `lora_alpha_ratio`.
  - Support explicit `lora_alpha` or positive fractional ratio (e.g. 0.5) to support rank 64 / alpha 32.
  - Preserve effective alpha in checkpoint metadata and reload validation.
  - Verify: training and serving load rank 64 / alpha 32 and agree on logprobs within measured tolerance. (Merged in #3)

---

## Phase 1: Core Trainer Correctness (P1)
- [x] **C3: Pre-backward Loss Validation & Rejection** (`src/tuft/backends/fsdp_engine.py`, `src/tuft/backends/fsdp_training_backend.py`)
  - Stop truncating oversized rows silently.
  - Stop substituting current-policy logprobs when behavior logprobs are missing.
  - Stop defaulting missing RL advantages to zero.
  - Validate required fields, exact per-row lengths, and finite values upfront before backward pass or gradient accumulation. Reject malformed requests immediately with clear client errors.
  - Verify: a malformed row in a later microbatch fails before any gradient accumulation. (Merged in #4)
- [x] **C4: FSDP Uneven Batch Schedule & Collective Sync** (`src/tuft/backends/fsdp_training_backend.py`)
  - Balance microbatch execution across DP ranks by padding shorter shards with zero-weight dummy datums (`create_zero_weight_dummy_datum`).
  - Ensure all DP actors execute the identical number of collective communication steps without dropping `micro_batch_size`.
  - Strip dummy datum outputs from the final response so client receives exactly the expected row outputs.
  - Verify: uneven batches preserve the microbatch limit, complete collectives without hanging, and match reference updates. (Merged in #6)
- [x] **C5: Distributed Actor Lifecycle & Robust Init** (`src/tuft/backends/fsdp_training_backend.py`)
  - Ensure partial Ray actor initialization failures cleanly terminate all locally created actors.
  - Bound controller wait times for worker group initialization.
  - On rank failure, fail the group and all pending work cleanly without replaying ambiguous optimizer steps.
  - Verify: inject failure at each initialization stage; verify no orphaned actors remain and retries succeed. (Merged in #4)

---

## Phase 2: Serving Lifecycle & LoRA Hot-Swap (P1)
- [x] **C1: Immutable Adapter Versioning & Reload Sync** (`src/tuft/oai/model_resolver.py`, `src/tuft/oai/router.py`, `src/tuft/sampling_controller.py`)
  - Use immutable version-specific adapter IDs (`f"{training_run_id}:{checkpoint_id}"`) so different checkpoints of the same training run never collide in routing caches.
  - Switch routing only after all inference replicas acknowledge the load.
  - Verify: publish checkpoint A then B; repeat cached prompt; verify served weights, captured version, and failure behavior on one replica. (Merged in #5)
- [x] **C2: Session Eviction & Backend Adapter Release** (`src/tuft/sampling_controller.py`, `src/tuft/state.py`)
  - Call backend adapter removal (`remove_adapter()`) when sampling sessions are evicted or unreferenced after active requests finish.
  - Preserve base model weights and checkpoint files while freeing GPU memory.
  - Verify: release one of two adapters; prove the other still works and released capacity is reusable; test attach-versus-reap races. (Merged in #5)

---

## Phase 3: Compatibility & Deployment Topology (P1)
- [x] **D2: Tinker Protobuf SDK & Cookbook Qualification**
  - Range-pinned `tinker>=0.25,<0.29` and qualified against both wire generations
    (0.25.0 and 0.28.1) in CI (`.github/workflows/checks.yml::tinker-compat`).
  - Wire conversion stays at the API boundary in `src/tuft/compat.py`, which now
    absorbs the drift found across the range:
    - 0.26.2 made sample sequence identity mandatory (`SampledSequence.sequence_id`
      and `UntypedAPIFuture.sample_sequence_ids`); `/api/v1/asample` now mints one
      id per requested sample so `SamplingClient.sample()` stops asserting.
    - 0.26.2 added `loss_fn_config_v2` (number|text); decoding prefers it and keeps
      the numeric-only contract (a string kwarg is a 422).
    - 0.28.0 renamed the top-k prompt-logprobs `prompt_length` field to `length`.
  - Verified end to end over the real SDK (`tests/test_tinker_sdk_e2e.py`, CPU/dummy
    backends): create adapter → forward/backward → optim_step → save → create sampler
    → sample, plus JSON-vs-protobuf retrieve wire formats, top-k round trip,
    server-minted sample sequence ids, pipelined out-of-order retrieval, duplicate /
    gapped seq_id rejection, and terminal-failure surfacing without hangs.
- [ ] **D3: OpenAI Proxy & AGL Rollout Capture** (`src/tuft/oai/proxy.py`)
  - Verify chosen token logprobs, token IDs, routing, and served-version capture match AGL requirements.
  - Verify: run AGL capture checks before and after publication, including in-flight requests.
- [x] **D1: Component Resource Groups & Placement** (`src/tuft/backends/sampling_backend.py`, `src/tuft/backends/fsdp_training_backend.py`)
  - `ModelConfig.train_gpu_resource` / `infer_gpu_resource` name Ray custom resources;
    every training actor requests 1 unit per GPU and every vLLM actor 1 unit per GPU
    (`ModelConfig.actor_resources`). Unset (default) requests nothing, so local dev
    and colocation schedule exactly as before. Nodes are labelled with
    `ray start --resources '{"train_gpu": N, "infer_gpu": M}'`.
  - Remaining: qualify on hardware — both target layouts (1 node: 6 train / 2 TP
    inference; multi-node: 4 train nodes + 1 inference node). An unsatisfiable
    resource name currently leaves the actor pending in the Ray scheduler instead of
    failing fast.
- [x] **D4: LoRA Adapter Staging on Inference Nodes** (`checkpoints.py`,
  `backends/vllm_engine.py`, `backends/sampling_backend.py`, `oai/router.py`)
  - Design: `../llmqa-team-junk/docs/research/2026-09-14-lora-weight-streaming.md`.
  - vLLM nodes need no shared filesystem: adapter bytes travel as a plain Ray arg and
    `VLLMEngine.stage_adapter` writes them under `/dev/shm/tuft-adapters-{uuid}`.
  - `VLLMSamplingBackend` owns both namespaces: sampling sessions (`add_adapter`) and
    OAI names (`ensure_oai_lora_loaded`). An OAI name is unstaged only after vLLM
    confirms the unload (2xx/404).
  - Idle TTL (`adapter_idle_ttl_minutes`, default 30, 0 = off) unloads and unstages
    idle adapters; the next request re-adds them. `max_loras` default 1 → 8, and it
    also caps staged sampling sessions (LRU). Adapters with requests in flight are never unloaded.
  - The server holds `LoraRef` tuples; `VLLMEngine` builds vLLM's `LoRARequest`, so the
    server runs in `tuft-train` (no vLLM). 2-node smoke passed 2026-09-29 (`2dc0a62`).
  - Deferred: trainer-side save/resume without shared `checkpoint_dir` (multi-node
    FSDP needs it anyway; revisit save and load together), disk GC of checkpoints.

---

## Phase 5: Shared-Server Hardening (P1, 2026-10-03 review)
Source: codebase audit + ecosystem research (tinker SDK 0.32.0, cookbook needs >=0.30.1, SkyRL, OpenRL).
- [x] **Upstream sync v0.2.1** (merge `27fbcee`): seq-guard fast-forward after a failed op (#162),
  chunked target logprobs, 1800 s proxy timeout, replicated small FSDP requests (#164), bounded
  micro-batches (#165). Dropped the fork's dummy-datum padding (`dummy_datum.py`) for upstream's
  zero-loss rounds. Kept the fork's publish workflow (no PyPI release from the fork).
- [x] **E1: tinker SDK 0.30+** — pin is `tinker>=0.25,<0.33`; `optim_step` reads `optim_params`,
  unsupported optimizer/sampling features return 400; CI matrix is 0.25.0 + 0.32.0.
  `sessions/{id}/finish` comes with E3.
- [ ] **E2: Tenant isolation** — OAI `resolve_model` gets the caller and applies owner/public checks,
  no silent base-model fallback on a missing adapter, `retrieve` by a non-owner must not mutate the future.
- [x] **E3: Resource release** — `POST /sessions/{id}/finish` and a heartbeat-TTL sweep
  (`session_heartbeat_ttl_minutes`, 30) release the session's runs (free slot, keep checkpoints, 410 on
  heartbeat); the same sweep drops completed futures older than `future_ttl_seconds` (now 1 h).
- [x] **E4: Durable checkpoints** — checkpoints resolve from disk after a restart without Redis
  (load, delete, publish, weights_info, archive); the archive route redirects to a signed HTTP tar download.
- [ ] **E5: Readiness + compat matrix** — `/readyz` that pings backends; committed per-method SDK
  compatibility table enforced by a CPU test.

---

## Phase 4: Measured Improvements (P2 - After Measurement)
- [ ] **Token-budget batching**: Length sorting and padded-token budget with coordinated FSDP microsteps.
- [ ] **Bounded fair admission**: Bounded turns across adapter queues; handle sequence gaps, duplicates, and cancellations.
- [ ] **Sharded model initialization**: Meta initialization and distributed materialization when base model exceeds single GPU memory during startup.
- [ ] **Replication versus sharding**: Benchmark replicated training when model fits on single H100; add mesh options only after measurement.
- [ ] **Tinker 0.29+ features**: top-k sample logprobs, target/alt prompt tokens, Dimuon optimizer (return 400 today).
- [ ] **Adapter capability reporting**: Expose supported rank/target geometry and free slot capacity; maintain static FSDP slot pool.
