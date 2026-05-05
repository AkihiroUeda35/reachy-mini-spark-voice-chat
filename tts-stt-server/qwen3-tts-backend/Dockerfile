ARG NGC_VLLM_IMAGE=nvcr.io/nvidia/vllm:26.04-py3
FROM ${NGC_VLLM_IMAGE}

ARG VLLM_VERSION=0.20.0
ARG VLLM_OMNI_REF=main

ARG VLLM_OMNI_RUNTIME_DEPS="av>=14.0.0 omegaconf>=2.3.0 diffusers>=0.36.0 accelerate==1.12.0 cache-dit==1.3.0 torchsde>=0.2.6 openai-whisper>=20250625 imageio[ffmpeg]>=2.37.2 x-transformers>=2.12.2 prettytable>=3.8.0 aenum==3.1.16 janus>=1.0.0 pydub onnxruntime>=1.23.2"

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip uninstall -y vllm \
    && python -m pip install --no-cache-dir "vllm==${VLLM_VERSION}" \
    && python -m pip install --no-cache-dir ${VLLM_OMNI_RUNTIME_DEPS} \
    && python -m pip install --no-cache-dir --no-deps "git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_REF}"

COPY qwen3-tts-backend/run.sh /usr/local/bin/qwen3-tts-run
RUN chmod +x /usr/local/bin/qwen3-tts-run

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

WORKDIR /workspace
ENTRYPOINT ["qwen3-tts-run"]