# Roadmap

Fork `k4black/fork-TuFT` of `agentscope-ai/TuFT`: an open-source server for the Tinker fine-tuning
API. Goal: a public, self-hosted Tinker that runs the tinker-cookbook unchanged and grows from one
GPU worker into a multi-model fleet on Kubernetes.

Branches: `main` is the integration branch and PR target; `upstream` tracks `agentscope-ai/TuFT:main`
for syncs (sync in only). Every change is a minimal diff in its own worktree; non-trivial features get
a design pass first. Per-method SDK status lives in `docs/compatibility.md`.

Priority: P1 = blocks a real user, P2 = needed for a shared server, P3 = measure first.
Phases are a default order, not a dependency chain.

Vocabulary
- **Worker**: one server process serving one base model: training (one replica of K nodes) and/or
  sampling (vLLM). Knows adapters and the base model, nothing about users.
- **Pool**: all workers for one base model and role. **Run**: one adapter under training, pinned to
  the worker that holds its optimizer state. **Adapter**: a saved LoRA; loadable on any sampling
  worker of its base model.
- **Control plane**: routes, scales and meters workers; validates identity from an external provider.

---

## Open work

### Phase 6: Tinker API parity (P1)
Done when the cookbook core recipes (SL, RL, DPO, distillation, multi-turn) run unchanged on the
current SDK.

- [ ] **Checkpoint TTL and disk GC** — honour `ttl_seconds` on `save_state` and
  `save_weights_for_sampler` and `PUT checkpoints/{id}/ttl`; store `expires_at` in `metadata.json`;
  a sweep removes expired records, files and staged copies; cap or make transient the unnamed
  `checkpoint-NNNN` sampler saves; report `get_current_checkpoint_storage_usage`.
- [ ] **`copy_weights`** — storage-only copy of a checkpoint into a new non-trainable run with
  the same checkpoint kind. Also the import path for external adapters: `source_path` accepts
  `hf://<repo>` and `s3://<bucket>/<key>` PEFT adapters, `weights_access_token` carries an HF token
  or presigned URL (worker env as fallback); the adapter is validated against the base model, rank,
  alpha and target modules and rejected on mismatch; the response is a `tinker://` path usable by
  `create_sampling_client` and `create_training_client_from_state`.
- [ ] **External export** — `save_weights_external` and `get_external_weights_urls` in the HF
  adapter layout the cookbook `weights/` module reads.
- [ ] **`topk_sample_logprobs`** — top-k logprobs at generated positions through vLLM and the
  proto encoder.
- [ ] **`target_prompt_logprobs`** — score chosen token ids at prompt positions, including sparse
  CSR input.
- [ ] **`prompt_alt_tokens_k`** — k alternative draws per prompt position from the same prefill.
- [ ] **Model metadata** — `get_info.model_data.arch` from the model config; a resolvable
  `tokenizer_id` for local paths and aliases so `get_tokenizer` works.
- [ ] **Telemetry and per-step metrics** — keep SDK telemetry events; expose per-step training
  metrics for W&B or MLflow.
- [ ] **Save-request fields dropped today** — `user_metadata` on saves, returned in checkpoint
  records; `LoraConfig.seed` for reproducible adapter init on both backends, including reused FSDP
  slots.
- [ ] **Adapter capability reporting** — advertise supported ranks (`fsdp_rank_slots`), target
  geometry and alpha rules in `get_server_capabilities` so clients fail before submitting.
- [ ] **CPU mode for tests** — the HF backend on CPU with a tiny model, compared against a plain
  transformers forward; vLLM on its CPU backend with a `TuFTCPUWorker` twin of the prompt-logprobs
  patch and a `device: cpu` engine mode, so sampling, LoRA and logprob paths run without a GPU.
- [ ] **Restart durability on a persistent volume** — `create_training_client_from_state` after
  a server restart with checkpoints on a PVC.
- [ ] **Keep as 400/404** — Dimuon optimizer, `assign_session_project`, `export_session_trace`,
  `get_audit_log`, `get_billing_usage`; noted in the compatibility table.

### Phase 7: The worker (P2)
A worker is one base model and knows no users; anyone who can reach it on the network can use it.
Identity, quotas and routing belong to the control plane (Phase 9).

- [ ] **Remove users from the worker** — drop API keys, owner checks and per-user state; one
  optional static service token as the only gate; sessions stay as a pure lifecycle mechanism
  (heartbeat, finish, release).
- [ ] **One base model per worker** — a config holds one model; multi-model configs remain only
  for colocated single-GPU development.
- [ ] **`/stats` and `/metrics`** — JSON `/stats` for the control plane: base model, loaded
  adapters, slots used and free, queue depth and wait, per-adapter GPU-seconds held and tokens
  trained and sampled, drain state. The same as Prometheus series on `/metrics`.
- [ ] **Drain** — `POST /admin/drain`: refuse new runs, finish the in-flight step, `save_state`
  every live run, release adapters, report done, exit.
- [ ] **Bounded admission and request dedup** — per-adapter queues with bounded turns; durable
  rejection of duplicate `(model_id, seq_id)`; explicit cancellation.
- [ ] **Packaging** — Helm chart or kustomize for one worker with `/api/v1/readyz`, a long startup
  probe (socket opens only after vLLM init), Docker `HEALTHCHECK`, `train_gpu`/`infer_gpu` Ray
  resources for the split layout.
- [ ] **Small fixes** — `release_run` logs nothing; `evict_session` sends one
  `unload_lora_adapter` per already-unloaded sampler; document that the default FSDP slot set is
  rank 8 plus `max_lora_rank` and that `fsdp_rank_slots` overrides it.
- [ ] **OpenAI proxy rollout capture** — chosen-token logprobs, token ids, routing and
  served-version capture verified before and after a publish, including in-flight requests.

### Phase 8: Single-worker scale (P3, measure first)
- [ ] **Multi-node FSDP** — shared checkpoint dir or trainer-side save and resume over Ray; one
  training replica spanning nodes.
- [ ] **Vision inputs** — `ImageChunk` already decodes; add the HF/FSDP and vLLM paths and run
  the cookbook `vlm_classifier` recipe.
- [ ] **Audio inputs** — investigation; needs preprocessing and model support beyond vision.
- [ ] **Mixed-adapter forwards** — several adapters in one training forward on one base.
- [ ] **Faster adapter sync** — delta or in-memory transport instead of a full adapter stage per
  save.
- [ ] **Token-budget batching** — length sorting and a padded-token budget with coordinated FSDP
  microsteps.
- [ ] **Sharded model init** — meta init and distributed materialization when the base model
  exceeds one GPU.
- [ ] **Replication vs sharding** — benchmark replicated training when the model fits one GPU
  before adding mesh options.
- [ ] **Sequence packing and sequence parallelism** — padding-free packed micro-batches and
  context/sequence parallel attention for long sequences (SkyRL and Twinkle have both).
- [ ] **CPU-offloaded adapter state** — keep inactive adapters and their optimizer state in pinned
  CPU memory and swap on demand, so a worker holds more adapters than its GPU slots (SkyRL pattern).
- [ ] **Hybrid LoRA with full-weight modules** — LoRA on most layers plus selected modules trained in
  full (embeddings, unembed, norms), beyond the Tinker `train_unembed` flag (Twinkle has it natively).
- [ ] **Tensor, pipeline and expert parallelism** — a Megatron-style backend for models that do not
  fit FSDP well (large MoE); only with a target model (SkyRL and Twinkle have Megatron paths).
- [ ] **Full fine-tuning** — optional; the Tinker API is LoRA-only.

### Phase 9: Control plane (P2, after Phase 7)
Many workers behind one control plane. Mechanics (pinning policy, scale-down choice, pool
definition, schema) are decided in a design pass when the phase starts.

- [ ] **Identity** — tokens validated against an external identity provider (static keys or
  JWT); the control plane keeps quotas and usage per subject and manages no accounts.
- [ ] **Ownership and routing** — records run → worker, sampler → pool, future → worker,
  checkpoint → storage; `create_model` routes by base model to a pool and picks a worker; later
  calls resolve by that map. Sessions and heartbeats move here. Dedup of `(model_id, seq_id)` moves here from the worker.
- [ ] **State store** — Postgres (SQLite for development) for sessions, futures, ownership and
  usage; defined recovery on restart and on worker loss (fail its futures, mark its runs released).
- [ ] **Checkpoint storage** — S3-compatible object storage behind `checkpoints.py`; 's
  logical TTL stays the authority, a bucket lifecycle rule is the backstop; sampling workers pull
  adapters by immutable `{run}:{checkpoint}` id.
- [ ] **Quotas and fair share** — per-subject run and in-flight caps; bounded turns per subject
  per pool, then a queue; 429 with `Retry-After` only when the queue is full. Usage attributed from
  worker per-adapter counters (`/stats`).
- [ ] **Standalone sampling workers** — `tuft-infer` as a plain vLLM server over HTTP with
  dynamic LoRA load and the prompt-logprobs patch; Ray only inside a training worker.
- [ ] **Drains in the fleet** — sampling drain (stop accepting, wait for in-flight, unload) and
  training drain (see Phase 7) driven by the control plane; a lost worker fails its futures fast.
- [ ] **Scaling signals** — per-pool metrics (slot occupancy, queue depth, tokens/s, idle time)
  for an external autoscaler; scale-to-zero only for sampling pools.
- [ ] **Adapter-aware sampling routing** — route a sample to the worker that already holds the
  adapter, per-worker adapter cap, least-loaded fallback with on-demand reload.
- [ ] **Fleet packaging** — control plane Deployment, pools as Helm values (base model, role,
  replicas, nodes per replica, GPU labels), drain hooks as `preStop`; kind-based tests with dummy
  workers.

### Phase 10: Dynamic fleet (P3)
- [ ] **Pools on demand** — create a training pool when a run for an unserved base model arrives;
  tear it down when idle.
- [ ] **Failure recovery** — detect a lost worker within one sweep; checkpoint on SIGTERM; resume a
  run on another worker from the last checkpoint, first with `load_state`, later transparently.
- [ ] **Exactly-once optimizer steps across crashes** — journal `(model_id, seq_id)` with the
  checkpoint it belongs to, so a replayed or duplicated request after a crash never applies an
  update twice (no reviewed server has this).
- [ ] **Elastic multi-node recovery** — re-form a training replica after a node loss and continue
  from the last checkpoint without a full restart.
- [ ] **Cluster provisioning** — scheduler-driven node and GPU allocation for pools on demand
  (OpenRL uses Kubernetes DRA and time-slicing).
- [ ] **Billing and audit** — per-subject GPU-seconds and tokens, `get_billing_usage`,
  `get_audit_log`.

### Where this stands against other Tinker servers
"?" = not visible from public code or the SDK.

| Capability | Hosted Tinker | OpenRL | SkyRL | Twinkle | This fork |
|---|---|---|---|---|---|
| Core train/sample/save/load | yes | SDK 0.29 | SDK 0.25 | SDK 0.16 via translation | SDK 0.25–0.32 |
| Client-defined custom losses | yes | no | partial | ? | yes |
| Session finish, release, heartbeat 410 | yes | partial | partial | ? | yes |
| Archive download | signed | roadmap | unsigned redirect | checkpoint service | signed, expiring |
| Checkpoint TTL and expiry | yes | roadmap | no | ? | Phase 6 |
| Import external adapters | no | no | no | no | Phase 6 |
| copy_weights, external export | yes | no | export only | ? | Phase 6 |
| Top-k sample and target prompt logprobs | yes | no | no | ? | Phase 6 |
| Backend readiness probe | ? | API health only | API health only | ? | yes |
| Compat matrix in CI | n/a | yes | no | no | yes |
| User-free worker | ? | yes | yes | no (token-aware) | Phase 7 |
| Worker stats and drain | ? | no | sample drain | queue limits | Phase 7 |
| Helm / Kubernetes-native | n/a | yes (DRA) | no | no (Ray Serve) | Phase 7 |
| Prometheus / profiling | ? | yes | profiler API | OTEL metrics | Phase 7 |
| Quotas and fair share | yes | roadmap | no | per-token rate limits | Phase 9 |
| External identity | yes | no auth | no auth | accepts any token | Phase 9 |
| Standalone sampling workers | ? | yes | external vLLM | separate sampler service | Phase 9 |
| Adapter-aware routing | ? | no | session affinity | ? | Phase 9 |
| Multiple LoRA ranks in one runtime | ? | padded to max rank | one rank and alpha | backend-dependent | yes (rank pools) |
| CPU-offloaded inactive adapters | ? | no | yes | ? | Phase 8 |
| Mixed-adapter training forward | ? | no | no | no | Phase 8 |
| Multi-node FSDP | yes | no | yes | framework-level | Phase 8 |
| Tensor / pipeline / expert parallelism | yes | no | Megatron | Megatron | Phase 8 |
| MoE training | yes | roadmap | Megatron MoE | expert-parallel adapters | routed-expert LoRA |
| Sequence packing, sequence parallelism | ? | chunked logprobs only | yes | yes | Phase 8 |
| Hybrid LoRA with full-weight modules | train_unembed only | no | no | yes | Phase 8 |
| Vision inputs | yes | no | yes | ? | Phase 8 |
| Full fine-tuning | no | yes | yes | yes | Phase 8 |
| Exactly-once optimizer steps across crashes | ? | no | no | no | Phase 10 |
| Elastic multi-node recovery | ? | no | no | no | Phase 10 |
| Cluster provisioning on demand | ? | DRA scheduler | no | no | Phase 10 |

---

## Done

### Shared-server hardening
- [x] Upstream sync v0.2.1: seq-guard fast-forward after a failed op, chunked target logprobs,
  1800 s proxy timeout, replicated small FSDP requests, bounded micro-batches.
- [x] tinker SDK 0.25 through 0.32 (#24): `optim_params` rename absorbed; unsupported optimizer and
  sampling fields return 400; CI runs both ends of the range.
- [x] Tenant isolation (#21): one shared `require_access` for the tinker and OpenAI paths; missing
  adapter is 404, not base-model fallback; foreign `retrieve` no longer mutates a future.
- [x] Resource release (#25): `POST /sessions/{id}/finish`; heartbeat-TTL sweep releases runs
  (adapter freed, checkpoints kept, 400 on reuse, 410 on heartbeat); completed futures evicted
  after a TTL.
- [x] Durable checkpoints (#22): checkpoints resolve from disk after a restart without Redis; signed
  expiring HTTP tar download; path-traversal guard; app-wide error mapping.
- [x] Readiness and compat matrix (#23): `/api/v1/readyz` pings vLLM and training actors;
  `docs/compatibility.md` enforced in CI.
- [x] GPU qualification on a 2-node split (hf and fsdp backends, SDK 0.25 and 0.32): readiness,
  finish → release, slot reuse, archive download via SDK and CLI.

### Compatibility and deployment topology
- [x] Ray placement (#13): `train_gpu` / `infer_gpu` custom resources per model.
- [x] tinker SDK 0.25–0.28 qualification (#12): protobuf wire, `compat.py` absorbs drift.
- [x] Adapter staging on inference nodes (#14, #17, #18): adapter bytes over Ray to `/dev/shm`, no
  shared filesystem for vLLM nodes; idle TTL and `max_loras` LRU; server runs without vLLM.
- [x] Split images `tuft-train` / `tuft-infer` (#10, #16) for CUDA 12 and 13 on Docker Hub; CUDA
  graphs on by default (#20).

### Serving lifecycle
- [x] Immutable adapter ids `{run}:{checkpoint}` and reload sync (#5).
- [x] Backend adapter release on session eviction (#5).

### Trainer correctness
- [x] Pre-backward validation, no silent truncation or defaults (#4).
- [x] Uneven FSDP batches (#6; superseded by upstream's zero-loss rounds).
- [x] Actor lifecycle and robust init (#4).

### Quick wins
- [x] torch 2.13 / vLLM 0.28 alignment (#8, #15), CI lint and dependabot rules (#7).
- [x] Fractional `lora_alpha_ratio` and explicit `lora_alpha` (#3).
