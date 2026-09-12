# Roadmap & Fork TODO

Based on `../llmqa-team-junk/docs/research/2026-09-11-tuft-fork-review.md` and repository inspection.
Target branches:
- `main`: Active development & PR merge target.
- `upstream`: Tracking `agentscope-ai/TuFT:main`.

All implementations must follow `/ponytail` (minimal code, reuse existing patterns, no bloat) and use isolated git worktrees. Features that are not small fixes require plans and `/grill-me` alignment before implementation.

---

## Phase 0: Quick Wins, Versions & Small Fixes
- [ ] **Dependency & vLLM Alignment**
  - Verify pinned vLLM version compatibility (`vllm==0.24.0` in `pyproject.toml`) and native LoRA hot-swap semantics.
  - Verify SDK dependencies (`tinker>=0.25,<0.26`).
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
- [ ] **C4: FSDP Uneven Batch Schedule & Collective Sync** (`src/tuft/backends/fsdp_training_backend.py`)
  - *Requires plan + `/grill-me`*
  - First reject unsupported shapes that cannot preserve the configured microbatch limit.
  - Introduce bounded schedule with equal collective steps across DP ranks, injecting zero-weight dummy microsteps where needed.
  - Verify: uneven batches preserve the microbatch limit, complete collectives without hanging, and match reference updates.
- [x] **C5: Distributed Actor Lifecycle & Robust Init** (`src/tuft/backends/fsdp_training_backend.py`)
  - Ensure partial Ray actor initialization failures cleanly terminate all locally created actors.
  - Bound controller wait times for worker group initialization.
  - On rank failure, fail the group and all pending work cleanly without replaying ambiguous optimizer steps.
  - Verify: inject failure at each initialization stage; verify no orphaned actors remain and retries succeed. (Merged in #4)

---

## Phase 2: Serving Lifecycle & LoRA Hot-Swap (P1)
- [ ] **C1: Immutable Adapter Versioning & Reload Sync** (`src/tuft/oai/model_resolver.py`, `src/tuft/oai/router.py`, `src/tuft/sampling_controller.py`)
  - *Requires plan + `/grill-me`*
  - Use immutable version-specific adapter IDs so different checkpoints of the same training run never collide in routing caches.
  - Use native vLLM in-place reload. Switch routing only after all inference replicas acknowledge the load. Drain requests or guard prefix caching during update.
  - Verify: publish checkpoint A then B; repeat cached prompt; verify served weights, captured version, and failure behavior on one replica.
- [ ] **C2: Session Eviction & Backend Adapter Release** (`src/tuft/sampling_controller.py`, `src/tuft/state.py`)
  - *Requires plan + `/grill-me`*
  - Call backend adapter removal when sampling sessions are evicted or unreferenced after active requests finish.
  - Implement heartbeat-based cleanup of abandoned sessions with active-operation guards.
  - Preserve base model weights and checkpoint files while freeing GPU memory.
  - Verify: release one of two adapters; prove the other still works and released capacity is reusable; test attach-versus-reap races.

---

## Phase 3: Compatibility & Deployment Topology (P1)
- [ ] **D2: Tinker Protobuf SDK & Cookbook Qualification**
  - Pin and test against stock Tinker SDK client (`tinker>=0.25`).
  - Wire conversion stays at API boundary.
  - Verify: end-to-end client sequence: create adapter → forward/backward → optimizer step → save → create sampler → sample (including pipelining, duplicates, errors, reordered seq IDs).
- [ ] **D3: OpenAI Proxy & AGL Rollout Capture** (`src/tuft/oai/proxy.py`)
  - Verify chosen token logprobs, token IDs, routing, and served-version capture match AGL requirements.
  - Verify: run AGL capture checks before and after publication, including in-flight requests.
- [ ] **D1: Component Resource Groups & Placement** (`src/tuft/backends/sampling_backend.py`, `src/tuft/backends/fsdp_training_backend.py`)
  - *Requires plan + `/grill-me`*
  - Add component resource groups using native Ray placement and node constraints for disjoint train/inference GPUs.
  - Verify actual allocation on creation; clean up reservations on failure.
  - Verify: qualify two hosts first, then both target layouts (1 node: 6 train / 2 TP inference; multi-node: 4 train nodes + 1 inference node).

---

## Phase 4: Measured Improvements (P2 - After Measurement)
- [ ] **Token-budget batching**: Length sorting and padded-token budget with coordinated FSDP microsteps.
- [ ] **Bounded fair admission**: Bounded turns across adapter queues; handle sequence gaps, duplicates, and cancellations.
- [ ] **Sharded model initialization**: Meta initialization and distributed materialization when base model exceeds single GPU memory during startup.
- [ ] **Replication versus sharding**: Benchmark replicated training when model fits on single H100; add mesh options only after measurement.
- [ ] **Adapter capability reporting**: Expose supported rank/target geometry and free slot capacity; maintain static FSDP slot pool.
