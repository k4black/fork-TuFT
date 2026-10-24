# TuFT on the official vLLM image: vLLM, the HF and FSDP trainers and flash-attn.
# torch and vLLM stay as the base ships them; keep VLLM_IMAGE in step with the
# vllm pin in pyproject.toml (the constraints below fail the build on drift).
#
#   docker build -f docker/Dockerfile -t tuft .
#   docker build -f docker/Dockerfile -t tuft:cu12 --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129 .
#   docker run --gpus all --rm -p 10610:10610 -v <host_dir>:/data \
#       --entrypoint tuft tuft launch --port 10610 --config /data/tuft_config.yaml
ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${VLLM_IMAGE}

# Dependencies first, source last, so source edits keep this layer cached.
COPY ./pyproject.toml /opt/tuft/
RUN pip3 list --format=freeze | grep -iE '^(torch|vllm|triton)==' \
        > /tmp/base-pins.txt \
    && uv pip install --system --no-cache --constraints /tmp/base-pins.txt \
        -r /opt/tuft/pyproject.toml --extra backend --extra persistence

# flash-attn from a pinned prebuilt wheel; restore vLLM's nccl afterwards.
# causal-conv1d has no torch 2.13 wheel, so it builds from source with the base's nvcc.
# Its setup.py fixes the GPU arch list; the -dev packages it needs are removed after.
# The cu129 base ships a CUDA 13 torchcodec that breaks `import vllm`; TuFT decodes no media.
RUN set -eu; \
    nccl="$(pip3 list --format=freeze | grep -iE '^nvidia-nccl-cu[0-9]+==' || true)"; \
    tcuda="$(python3 -c 'import torch; print(torch.version.cuda)')"; \
    case "$tcuda" in \
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
    v="$(echo "$tcuda" | tr . -)"; dev=""; \
    for p in cuda-cudart-dev-$v libcusparse-dev-$v libcusolver-dev-$v; do \
      dpkg -s "$p" >/dev/null 2>&1 || dev="$dev $p"; done; \
    if [ -n "$dev" ]; then apt-get update -y; \
      apt-get install -y --no-install-recommends --allow-change-held-packages $dev; fi; \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=2 uv pip install --system --no-cache \
        --no-build-isolation --constraints /tmp/base-pins.txt causal-conv1d==1.7.0; \
    [ -z "$dev" ] || apt-get purge -y --autoremove $dev; \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/pip-*; \
    [ -z "$nccl" ] || uv pip install --system --no-cache --no-deps "$nccl"; \
    rm "/tmp/$whl"; \
    python3 -c "import flash_attn, fla, causal_conv1d as cc; print('flash_attn', flash_attn.__version__, 'causal_conv1d', cc.__version__)"

COPY ./scripts/verify_runtime_versions.py /opt/tuft/scripts/
COPY ./LICENSE ./README.md /opt/tuft/
COPY ./src /opt/tuft/src
RUN uv pip install --system --no-cache --no-deps /opt/tuft \
    && python3 /opt/tuft/scripts/verify_runtime_versions.py \
    && python3 -c "from tuft.backends.vllm_worker import TuFTGPUWorker; print('smoke OK')"
