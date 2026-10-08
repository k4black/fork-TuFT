"""Time an RL-like loop through a running TuFT server, for comparing colocate modes.

Each of K concurrent runs does N steps of forward_backward, optim_step,
save_weights_and_get_sampling_client and one grouped sample with prompt logprobs.
Prints a markdown table and writes JSON. Peak GPU memory comes from nvidia-smi.

    python scripts/bench_colocate.py --label sleep --runs 2 --steps 5 --json sleep.json
"""

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import tinker
from tinker import types


PHASES = ("forward_backward", "optim_step", "save_sampler", "sample")


def gpu_memory_poller(stop: threading.Event, peak: list[int]) -> None:
    """Track the peak summed ``memory.used`` (MiB) over all GPUs."""
    while not stop.wait(0.5):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        peak[0] = max(peak[0], sum(int(x) for x in out.split()))


def datum(rng: random.Random, length: int) -> types.Datum:
    tokens = [rng.randrange(100, 10000) for _ in range(length + 1)]
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=tokens[1:], dtype="int64", shape=[length]),
            "weights": types.TensorData(data=[1.0] * length, dtype="float32", shape=[length]),
        },
    )


def run(service: tinker.ServiceClient, args: argparse.Namespace, seed: int) -> dict:
    rng = random.Random(seed)
    trainer = service.create_lora_training_client(base_model=args.model, rank=args.rank)
    times: dict[str, list[float]] = {p: [] for p in PHASES}
    train_tokens = sample_tokens = 0

    def timed(phase: str, fn, *fn_args, **fn_kwargs):
        t = time.perf_counter()
        out = fn(*fn_args, **fn_kwargs)
        out = out.result() if isinstance(out, (Future, tinker.APIFuture)) else out
        times[phase].append(time.perf_counter() - t)
        return out

    for _ in range(args.steps):
        batch = [datum(rng, args.seq_len) for _ in range(args.batch)]
        timed("forward_backward", trainer.forward_backward, batch, "cross_entropy")
        timed("optim_step", trainer.optim_step, types.AdamParams(learning_rate=1e-5))
        sampler = timed("save_sampler", trainer.save_weights_and_get_sampling_client)
        prompt = [rng.randrange(100, 10000) for _ in range(args.prompt_len)]
        res = timed(
            "sample",
            sampler.sample,
            types.ModelInput.from_ints(prompt),
            args.group,
            types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0),
            include_prompt_logprobs=True,
        )
        train_tokens += args.batch * args.seq_len
        sample_tokens += sum(len(s.tokens) for s in res.sequences)
    return {"times": times, "train_tokens": train_tokens, "sample_tokens": sample_tokens}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--base-url", default=os.environ.get("TINKER_BASE_URL", "http://127.0.0.1:10610")
    )
    p.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-bench"))
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--label", default="", help="row name, e.g. the colocate mode")
    p.add_argument("--runs", type=int, default=1, help="concurrent training runs (K)")
    p.add_argument("--steps", type=int, default=5, help="steps per run (N)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group", type=int, default=8, help="samples per prompt")
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--json", help="write results here")
    args = p.parse_args()

    service = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    stop, peak = threading.Event(), [0]
    if shutil.which("nvidia-smi"):
        threading.Thread(target=gpu_memory_poller, args=(stop, peak), daemon=True).start()
    start = time.perf_counter()
    with ThreadPoolExecutor(args.runs) as pool:
        results = list(pool.map(lambda i: run(service, args, i), range(args.runs)))
    wall = time.perf_counter() - start
    stop.set()

    row = {
        "label": args.label,
        "runs": args.runs,
        "steps": args.steps,
        "wall_s": round(wall, 1),
        **{
            f"{ph}_s": round(statistics.mean(t for r in results for t in r["times"][ph]), 2)
            for ph in PHASES
        },
        "train_tok_s": round(sum(r["train_tokens"] for r in results) / wall),
        "sample_tok_s": round(sum(r["sample_tokens"] for r in results) / wall),
        "peak_gpu_mib": peak[0] or None,
    }
    print("| " + " | ".join(row) + " |")
    print("|" + "---|" * len(row))
    print("| " + " | ".join(str(v) for v in row.values()) + " |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "result": row}, f, indent=2)


if __name__ == "__main__":
    main()
