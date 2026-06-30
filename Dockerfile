FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 + Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    curl \
    vim \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# python3.11에 pip 설치 + 기본 python 심볼링크
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/bin/pip

WORKDIR /DataAnalysis_projects

# PyTorch 2.11 (Blackwell sm_120 지원) + CUDA 12.8
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 프로젝트 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter 커널 등록
RUN python -m ipykernel install --user --name pytorch --display-name "Python (PyTorch)"
