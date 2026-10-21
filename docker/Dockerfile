# TuFT image: the official vLLM server image plus TuFT. It is the only image.
# It holds vLLM, the HF and FSDP trainers (PyTorch, PEFT, Transformers, Ray) and
# flash-attn. vLLM worker processes import TuFT's prompt-logprobs patch
# (tuft.backends.vllm_worker.TuFTGPUWorker, wired in through the `worker_cls`
# engine argument set by src/tuft/backends/vllm_engine.py).
#
# CUDA, PyTorch and vLLM all come from the base image and are never reinstalled:
# TuFT pins exactly the versions it ships (vllm 0.32.0, torch 2.13.0), so the
# install below only adds TuFT's pure-Python dependencies.
#
# Build (CUDA 13, default):
#   docker build -f docker/Dockerfile -t tuft:cu13 .
#
# Build (CUDA 12.9, for driver 550 hosts):
#   docker build -f docker/Dockerfile -t tuft:cu12 \
#     --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129 .
#
# Run `tuft launch`. The base image ENTRYPOINT is vLLM's server, so pass
# `--entrypoint tuft` with `launch ...` as the command:
#   docker run --gpus all --rm -p 10610:10610 -v <host_dir>:/data \
#     --entrypoint tuft tuft:cu13 launch --port 10610 --config /data/tuft_config.yaml

# Keep in step with the vllm pin in pyproject.toml; the constraint below fails
# the build when they drift.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${VLLM_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /bin/uv

# Dependencies before source, so source edits keep this layer cached. The base
# image's torch and vLLM are frozen as constraints, so a pyproject.toml asking
# for other versions fails to resolve instead of reinstalling them.
COPY ./pyproject.toml /opt/tuft/
RUN pip3 list --format=freeze | grep -iE '^(torch|vllm|triton)==' \
        > /tmp/base-pins.txt \
    && uv pip install --system --no-cache --constraints /tmp/base-pins.txt \
        -r /opt/tuft/pyproject.toml --extra backend --extra persistence

# Colocated HF trainer kernels from pinned prebuilt wheels; then restore vLLM's nccl.
# The cu129 base ships a CUDA 13 torchcodec (it needs libnvrtc.so.13); vLLM tolerates
# a missing torchcodec but not a broken one, and TuFT decodes no audio or video.
RUN set -eu; \
    nccl="$(pip3 list --format=freeze | grep -iE '^nvidia-nccl-cu[0-9]+==' || true)"; \
    case "$(python3 -c 'import torch; print(torch.version.cuda)')" in \
      12.*) cuda=cu126; sha=41806f85fae421e0b0e70edde3121fd7126126aeba610fefa5d1b6cff88197af; \
            uv pip uninstall --system torchcodec ;; \
      13.*) cuda=cu130; sha=c65d762d24b7dd1ea8dbd9f5a98708fa8abbfd8f33f46bfc6123e76f1e174d3f ;; \
      *) echo "no pinned flash-attn wheel for this CUDA"; exit 1 ;; \
    esac; \
    whl="flash_attn-2.8.3+${cuda}torch2.13-cp312-cp312-linux_x86_64.whl"; \
    curl -fsSL -o "/tmp/$whl" \
      "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/$(echo "$whl" | sed 's/+/%2B/')"; \
    echo "$sha  /tmp/$whl" | sha256sum -c -; \
    uv pip install --system --no-cache --constraints /tmp/base-pins.txt \
        "/tmp/$whl" "fla-core>=0.5.0" "flash-linear-attention>=0.4.2"; \
    [ -z "$nccl" ] || uv pip install --system --no-cache --no-deps "$nccl"; \
    rm "/tmp/$whl"; \
    python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

COPY ./scripts/verify_runtime_versions.py /opt/tuft/scripts/
COPY ./LICENSE ./README.md /opt/tuft/
COPY ./src /opt/tuft/src
RUN uv pip install --system --no-cache --no-deps /opt/tuft \
    && python3 /opt/tuft/scripts/verify_runtime_versions.py \
    && python3 -c "from tuft.backends.vllm_worker import TuFTGPUWorker; print('smoke OK')"

# The base image's entrypoint (the vLLM OpenAI API server) is kept as is.
