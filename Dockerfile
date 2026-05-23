ARG NGC_VLLM_IMAGE=nvcr.io/nvidia/vllm:26.04-py3
ARG NGC_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3

FROM ${NGC_VLLM_IMAGE} AS tts-runtime

USER root

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

ARG TTS_TORCH_VERSION=2.11.0
ARG TTS_TORCHAUDIO_VERSION=2.11.0
ARG TTS_TORCHVISION_VERSION=0.26.0
ARG TTS_VLLM_VERSION=0.20.1
ARG VLLM_OMNI_REF=main
ARG VLLM_OMNI_RUNTIME_DEPS="av>=14.0.0 omegaconf>=2.3.0 diffusers>=0.36.0 accelerate==1.12.0 cache-dit==1.3.0 torchsde>=0.2.6 openai-whisper>=20250625 imageio[ffmpeg]>=2.37.2 x-transformers>=2.12.2 prettytable>=3.8.0 aenum==3.1.16 janus>=1.0.0 pydub onnxruntime>=1.23.2"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip uninstall -y vllm torch torchaudio torchvision \
    && python -m pip install --no-cache-dir \
        "torch==${TTS_TORCH_VERSION}" \
        "torchaudio==${TTS_TORCHAUDIO_VERSION}" \
        "torchvision==${TTS_TORCHVISION_VERSION}" \
    && python -m pip install --no-cache-dir --no-deps "vllm==${TTS_VLLM_VERSION}" \
    && python -m pip install --no-cache-dir ${VLLM_OMNI_RUNTIME_DEPS} \
    && python -m pip install --no-cache-dir --no-deps "git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_REF}"


ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

ENTRYPOINT ["qwen3-tts-run"]

FROM ${NGC_VLLM_IMAGE} AS stt-runtime

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


EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]

FROM ${NGC_PYTORCH_IMAGE} AS tsukasa-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
        mecab \
        mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        SoundFile \
        a_unet \
        accelerate \
        cutlet \
        einops \
        einops-exts \
        fastapi \
        fugashi \
        huggingface_hub \
        ipython \
        konoha \
        librosa \
        matplotlib \
        munch \
        nltk \
        openai \
        pydub \
        pyyaml \
        scipy \
        sentencepiece \
        tqdm \
        transformers==4.41.2 \
        typing-extensions \
        unidic-lite \
        uvicorn[standard] \
        xlstm \
    && python -m pip install --no-cache-dir --no-deps torchaudio==2.7.0 \
    && python -m pip install --no-cache-dir git+https://github.com/resemble-ai/monotonic_align.git

COPY tts/tsukasa_speech/run.sh /workspace/tts/tsukasa_speech/run.sh
COPY tts/tsukasa_speech/api.py /workspace/tts/tsukasa_speech/api.py

EXPOSE 5001

ENTRYPOINT ["sh", "/workspace/tts/tsukasa_speech/run.sh"]