ARG NGC_VLLM_IMAGE=nvcr.io/nvidia/vllm:26.04-py3

FROM ${NGC_VLLM_IMAGE} AS vllm-base

USER root

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

FROM vllm-base AS llm-runtime

# Keep torch aligned with spark-vllm-docker's runner image unless we intentionally diverge.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.11.0
# Override the wheel's dependency constraint locally for models that require Transformers 5.x.
ARG TRANSFORMERS_VERSION=5.7.0

RUN python -m pip uninstall -y vllm || true

RUN python -m pip install --no-cache-dir \
    --index-url ${TORCH_INDEX_URL} \
    "torch==${TORCH_VERSION}" \
    torchvision \
    torchaudio \
    triton

COPY spark-vllm-docker/wheels/*.whl /tmp/wheels/

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

FROM llm-runtime AS tts-runtime

ARG VLLM_OMNI_REF=main
ARG VLLM_OMNI_RUNTIME_DEPS="av>=14.0.0 omegaconf>=2.3.0 diffusers>=0.36.0 accelerate==1.12.0 cache-dit==1.3.0 torchsde>=0.2.6 openai-whisper>=20250625 imageio[ffmpeg]>=2.37.2 x-transformers>=2.12.2 prettytable>=3.8.0 aenum==3.1.16 janus>=1.0.0 pydub onnxruntime>=1.23.2"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir ${VLLM_OMNI_RUNTIME_DEPS} \
    && python -m pip install --no-cache-dir --no-deps "git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_REF}"

COPY tts/run.sh /usr/local/bin/qwen3-tts-run
RUN chmod +x /usr/local/bin/qwen3-tts-run

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

ENTRYPOINT ["qwen3-tts-run"]

FROM vllm-base AS stt-runtime

ARG CTRANSLATE2_VERSION=4.7.1
ARG CTRANSLATE2_CUDA_ARCHITECTURES=86
ARG CTRANSLATE2_CUDA_ARCH_LIST=8.6+PTX

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        libopenblas-dev \
        libsndfile1 \
        ninja-build \
        sox \
    && rm -rf /var/lib/apt/lists/*

COPY stt/requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip uninstall -y ctranslate2 \
    && git clone --recursive --branch v${CTRANSLATE2_VERSION} https://github.com/OpenNMT/CTranslate2.git /tmp/CTranslate2 \
    && cmake -S /tmp/CTranslate2 -B /tmp/CTranslate2/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CLI=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${CTRANSLATE2_CUDA_ARCHITECTURES} \
        -DCUDA_ARCH_LIST=${CTRANSLATE2_CUDA_ARCH_LIST} \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=OFF \
        -DWITH_MKL=OFF \
        -DWITH_OPENBLAS=ON \
        -DOPENMP_RUNTIME=COMP \
    && cmake --build /tmp/CTranslate2/build --parallel \
    && cmake --install /tmp/CTranslate2/build \
    && python -m pip install -r /tmp/CTranslate2/python/install_requirements.txt \
    && cd /tmp/CTranslate2/python \
    && CTRANSLATE2_ROOT=/usr/local python setup.py bdist_wheel \
    && pip install dist/*.whl \
    && rm -rf /tmp/CTranslate2

COPY stt/app ./app

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]