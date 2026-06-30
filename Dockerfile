FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /DataAnalysis_projects

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter 커널 등록
RUN python -m ipykernel install --user --name pytorch --display-name "Python (PyTorch)"
