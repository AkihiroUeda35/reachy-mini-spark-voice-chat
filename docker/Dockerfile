ARG NGC_VLLM_IMAGE=nvcr.io/nvidia/vllm:26.04-py3

FROM ${NGC_VLLM_IMAGE} AS vllm-base

USER root

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

FROM vllm-base AS llm-runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.11.0
ARG VLLM_WHEEL=spark-vllm-docker/wheels/vllm-0.20.1rc1.dev55+g3f1a4bb63.d20260429.cu132-cp312-cp312-linux_aarch64.whl
ARG FLASHINFER_PYTHON_WHEEL=spark-vllm-docker/wheels/flashinfer_python-0.6.9-py3-none-any.whl
ARG FLASHINFER_CUBIN_WHEEL=spark-vllm-docker/wheels/flashinfer_cubin-0.6.9-py3-none-any.whl
ARG FLASHINFER_JIT_CACHE_WHEEL=spark-vllm-docker/wheels/flashinfer_jit_cache-0.6.9-cp39-abi3-manylinux_2_28_aarch64.whl
ARG TRANSFORMERS_VERSION=5.7.0

RUN python -m pip uninstall -y vllm || true

RUN python -m pip install --no-cache-dir \
    --index-url ${TORCH_INDEX_URL} \
    "torch==${TORCH_VERSION}" \
    torchvision \
    torchaudio \
    triton

COPY ${VLLM_WHEEL} /tmp/wheels/
COPY ${FLASHINFER_PYTHON_WHEEL} /tmp/wheels/
COPY ${FLASHINFER_CUBIN_WHEEL} /tmp/wheels/
COPY ${FLASHINFER_JIT_CACHE_WHEEL} /tmp/wheels/

RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && python -m pip install --no-cache-dir "transformers==${TRANSFORMERS_VERSION}" \
    && python -m pip install --no-cache-dir fastsafetensors instanttensor \
    && rm -rf /tmp/wheels

COPY llm/run.sh /usr/local/bin/qwen-llm-run
RUN chmod +x /usr/local/bin/qwen-llm-run

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

ENTRYPOINT ["qwen-llm-run"]

FROM vllm-base AS tts-runtime

ARG VLLM_VERSION=0.20.0
ARG VLLM_OMNI_REF=main
ARG VLLM_OMNI_RUNTIME_DEPS="av>=14.0.0 omegaconf>=2.3.0 diffusers>=0.36.0 accelerate==1.12.0 cache-dit==1.3.0 torchsde>=0.2.6 openai-whisper>=20250625 imageio[ffmpeg]>=2.37.2 x-transformers>=2.12.2 prettytable>=3.8.0 aenum==3.1.16 janus>=1.0.0 pydub onnxruntime>=1.23.2"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip uninstall -y vllm || true \
    && python -m pip install --no-cache-dir "vllm==${VLLM_VERSION}" \
    && python -m pip install --no-cache-dir ${VLLM_OMNI_RUNTIME_DEPS} \
    && python -m pip install --no-cache-dir --no-deps "git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_REF}"

COPY tts/run.sh /usr/local/bin/qwen3-tts-run
RUN chmod +x /usr/local/bin/qwen3-tts-run

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

ENTRYPOINT ["qwen3-tts-run"]

FROM python:3.12-slim AS stt-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY stt/requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY stt/app ./app

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]