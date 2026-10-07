# Tinker SDK compatibility

One row per public method of the Tinker SDK clients (checked against `tinker==0.32.0`; CI also runs it on 0.25.0).
An `_async` twin has the same status as its sync method.
`tests/test_compat_matrix.py` fails when the SDK gains a method without a row here, or when a
row's route disagrees with the routes the server registers.
That test checks method names against registered routes only. `tests/test_tinker_sdk_e2e.py` covers
the behaviour of the core flow.

Status values:

- `supported`: the route exists, and the e2e or unit tests exercise it.
- `partial`: the route exists, but it answers with placeholder data or drops part of the request (see the note).
- `unsupported`: the server has no route for the call. The method fails with 404.
- `client-side`: the method makes no request of its own, or only composes other rows.

The route column names the main request. Methods that compose several requests list the last one.

| Method | Status | Route | Note |
|---|---|---|---|
| `ServiceClient.close` | supported | POST /api/v1/sessions/{session_id}/finish | Marks the session finished and releases its runs; later heartbeats get 410. |
| `ServiceClient.copy_weights` | supported | POST /api/v1/copy_weights | `tinker://` copies hard-link the files and keep the kind. `hf://org/repo[@rev][/subdir]` (request token only) and `s3://` (under `import_s3_prefixes`) import safetensors adapters as sampler checkpoints. The adapter must match a configured model (`base_model_name_or_path`), a supported rank, the server alpha rule (PEFT's usual `lora_alpha == r` needs `lora_alpha_ratio: 1`) and the targets of some `train_attn/train_mlp/train_unembed` combination, so a `q_proj,v_proj`-only adapter is rejected. The copy lives in a new released run: load it into a new run to train. |
| `ServiceClient.create_lora_training_client` | supported | POST /api/v1/create_model | |
| `ServiceClient.create_rest_client` | client-side | - | |
| `ServiceClient.create_sampling_client` | supported | POST /api/v1/create_sampling_session | |
| `ServiceClient.create_training_client_from_state` | supported | POST /api/v1/load_weights | Also calls `weights_info` and `create_model`. |
| `ServiceClient.create_training_client_from_state_with_optimizer` | supported | POST /api/v1/load_weights | Also calls `weights_info` and `create_model`. |
| `ServiceClient.get_console_url` | client-side | - | Links to the Thinking Machines console. TuFT has no console. |
| `ServiceClient.get_server_capabilities` | supported | GET /api/v1/get_server_capabilities | |
| `ServiceClient.get_telemetry` | partial | POST /api/v1/telemetry | The server accepts telemetry events and discards them. |
| `TrainingClient.create_sampling_client` | supported | POST /api/v1/create_sampling_session | |
| `TrainingClient.forward` | supported | POST /api/v1/forward_backward | Sends `forward_only=true`. |
| `TrainingClient.forward_backward` | supported | POST /api/v1/forward_backward | |
| `TrainingClient.forward_backward_custom` | supported | POST /api/v1/forward_backward | The SDK computes the custom loss between a forward and a backward call. |
| `TrainingClient.get_console_url` | client-side | - | Links to the Thinking Machines console. |
| `TrainingClient.get_info` | partial | POST /api/v1/get_info | `model_data.arch` is hard-coded to `toy-transformer`. |
| `TrainingClient.get_telemetry` | partial | POST /api/v1/telemetry | The server accepts telemetry events and discards them. |
| `TrainingClient.get_tokenizer` | partial | POST /api/v1/get_info | Loads `tokenizer_id` from the HF Hub. TuFT sets it to the configured `model_name`. |
| `TrainingClient.load_state` | supported | POST /api/v1/load_weights | |
| `TrainingClient.load_state_with_optimizer` | supported | POST /api/v1/load_weights | 400 for a sampler checkpoint: it holds no optimizer state. |
| `TrainingClient.optim_step` | supported | POST /api/v1/optim_step | |
| `TrainingClient.save_state` | supported | POST /api/v1/save_weights | Stores `ttl_seconds` and `user_metadata`. A save under an existing name replaces it; `overwrite` is ignored. |
| `TrainingClient.save_weights_and_get_sampling_client` | supported | POST /api/v1/create_sampling_session | Calls `save_weights_for_sampler` first. The server keeps the newest `sampler_checkpoints_keep` unnamed saves per live run and deletes all of them on release; older clients get 404. |
| `TrainingClient.save_weights_external` | unsupported | POST /api/v1/save_weights_external | |
| `TrainingClient.save_weights_for_sampler` | supported | POST /api/v1/save_weights_for_sampler | Stores `ttl_seconds` and `user_metadata`. |
| `SamplingClient.compute_logprobs` | supported | POST /api/v1/asample | Samples one token with prompt logprobs. |
| `SamplingClient.create` | supported | POST /api/v1/create_sampling_session | |
| `SamplingClient.from_sampler_handle` | client-side | - | 0.32+; rebuilds a client from a handle string. |
| `SamplingClient.get_base_model` | supported | GET /api/v1/samplers/{sampler_id} | |
| `SamplingClient.get_sampler_handle` | client-side | - | 0.32+; serialises the client without credentials. |
| `SamplingClient.get_telemetry` | partial | POST /api/v1/telemetry | The server accepts telemetry events and discards them. |
| `SamplingClient.get_tokenizer` | partial | GET /api/v1/samplers/{sampler_id} | Loads `base_model` from the HF Hub. TuFT sets it to the configured `model_name`. |
| `SamplingClient.on_queue_state_change` | client-side | - | |
| `SamplingClient.sample` | supported | POST /api/v1/asample | |
| `RestClient.assign_session_project` | unsupported | PUT /api/v1/sessions/{session_id}/project | |
| `RestClient.delete_checkpoint` | supported | DELETE /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id} | |
| `RestClient.delete_checkpoint_from_tinker_path` | supported | DELETE /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id} | |
| `RestClient.export_session_trace` | unsupported | GET /api/v1/sessions/{session_id}/trace_export | |
| `RestClient.get_audit_log` | unsupported | GET /api/v1/audit | |
| `RestClient.get_billing_usage` | unsupported | GET /api/v1/billing/usage/events | |
| `RestClient.get_current_checkpoint_storage_usage` | unsupported | GET /api/v1/billing/usage/checkpoints/current | 0.32+ |
| `RestClient.get_checkpoint_archive_url` | supported | GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive | |
| `RestClient.get_checkpoint_archive_url_from_tinker_path` | supported | GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive | |
| `RestClient.get_external_weights_urls` | unsupported | GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/external_weights_urls | |
| `RestClient.get_sampler` | supported | GET /api/v1/samplers/{sampler_id} | |
| `RestClient.get_session` | supported | GET /api/v1/sessions/{session_id} | |
| `RestClient.get_telemetry` | partial | POST /api/v1/telemetry | The server accepts telemetry events and discards them. |
| `RestClient.get_training_run` | supported | GET /api/v1/training_runs/{model_id} | |
| `RestClient.get_training_run_by_tinker_path` | supported | GET /api/v1/training_runs/{model_id} | |
| `RestClient.get_weights_info_by_tinker_path` | supported | POST /api/v1/weights_info | |
| `RestClient.list_checkpoints` | supported | GET /api/v1/training_runs/{model_id}/checkpoints | Reads `metadata.json` from disk when the run is not in memory (restart without Redis). |
| `RestClient.list_sessions` | supported | GET /api/v1/sessions | |
| `RestClient.list_training_runs` | supported | GET /api/v1/training_runs | |
| `RestClient.list_user_checkpoints` | supported | GET /api/v1/checkpoints | |
| `RestClient.publish_checkpoint_from_tinker_path` | supported | POST /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/publish | |
| `RestClient.set_checkpoint_ttl_from_tinker_path` | supported | PUT /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/ttl | Owner only. A 60 s sweep deletes expired checkpoints; it skips one a live sampler holds. |
| `RestClient.unpublish_checkpoint_from_tinker_path` | supported | DELETE /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/publish | |
| `RestClient.whoami` | client-side | - | Reads identity from the auth JWT. Raises against TuFT: the server disables JWT auth. |

## Numerics

The table covers the protocol only. `tests/tuft_mismatch_probe_real.py` measures the
training-vs-sampling logprob mismatch against a running server:

```bash
TINKER_API_KEY=... python tests/tuft_mismatch_probe_real.py \
    --base-url http://127.0.0.1:10610 \
    --model Qwen/Qwen3.5-4B \
    --tokenizer <local tokenizer path> \
    --tag <run tag> \
    --output /tmp/tuft_mismatch_real.json
```
