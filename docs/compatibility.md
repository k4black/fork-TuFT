# Tinker SDK compatibility

One row per public method of the Tinker SDK clients (checked against `tinker==0.28.1`).
An `_async` twin has the same status as its sync method.
`tests/test_compat_matrix.py` fails when the SDK gains a method without a row here, or when a
row's route disagrees with the routes the server registers.

Status values:

- `supported`: the server implements the route the method calls.
- `partial`: the route exists, but the server drops or fakes part of the behaviour (see the note).
- `unsupported`: the server has no route for the call. The method fails with 404.
- `client-side`: the method makes no request of its own, or only composes other rows.

The route column names the main request. Methods that compose several requests list the last one.

| Method | Status | Route | Note |
|---|---|---|---|
| `ServiceClient.close` | unsupported | POST /api/v1/sessions/{session_id}/finish | The SDK logs the 404 and still closes its local clients. |
| `ServiceClient.copy_weights` | unsupported | POST /api/v1/copy_weights | |
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
| `TrainingClient.load_state_with_optimizer` | supported | POST /api/v1/load_weights | |
| `TrainingClient.optim_step` | supported | POST /api/v1/optim_step | |
| `TrainingClient.save_state` | supported | POST /api/v1/save_weights | |
| `TrainingClient.save_weights_and_get_sampling_client` | supported | POST /api/v1/create_sampling_session | Calls `save_weights_for_sampler` first. |
| `TrainingClient.save_weights_external` | unsupported | POST /api/v1/save_weights_external | |
| `TrainingClient.save_weights_for_sampler` | supported | POST /api/v1/save_weights_for_sampler | |
| `SamplingClient.compute_logprobs` | supported | POST /api/v1/asample | Samples one token with prompt logprobs. |
| `SamplingClient.create` | supported | POST /api/v1/create_sampling_session | |
| `SamplingClient.get_base_model` | supported | GET /api/v1/samplers/{sampler_id} | |
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
| `RestClient.get_checkpoint_archive_url` | supported | GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive | |
| `RestClient.get_checkpoint_archive_url_from_tinker_path` | supported | GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/archive | |
| `RestClient.get_external_weights_urls` | unsupported | GET /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/external_weights_urls | |
| `RestClient.get_sampler` | supported | GET /api/v1/samplers/{sampler_id} | |
| `RestClient.get_session` | supported | GET /api/v1/sessions/{session_id} | |
| `RestClient.get_telemetry` | partial | POST /api/v1/telemetry | The server accepts telemetry events and discards them. |
| `RestClient.get_training_run` | supported | GET /api/v1/training_runs/{model_id} | |
| `RestClient.get_training_run_by_tinker_path` | supported | GET /api/v1/training_runs/{model_id} | |
| `RestClient.get_weights_info_by_tinker_path` | supported | POST /api/v1/weights_info | |
| `RestClient.list_checkpoints` | supported | GET /api/v1/training_runs/{model_id}/checkpoints | |
| `RestClient.list_sessions` | supported | GET /api/v1/sessions | |
| `RestClient.list_training_runs` | supported | GET /api/v1/training_runs | |
| `RestClient.list_user_checkpoints` | supported | GET /api/v1/checkpoints | |
| `RestClient.publish_checkpoint_from_tinker_path` | supported | POST /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/publish | |
| `RestClient.set_checkpoint_ttl_from_tinker_path` | unsupported | PUT /api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}/ttl | |
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
