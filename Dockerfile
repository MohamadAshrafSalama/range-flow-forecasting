FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml .
COPY environment.yml .

RUN pip install --upgrade pip && \
    pip install \
        nuscenes-devkit==1.2.0 \
        pyquaternion==0.9.9 \
        einops==0.8.1 \
        open3d==0.19.0 \
        opencv-python-headless==4.11.0.86 \
        matplotlib==3.9.4 \
        pillow \
        tqdm \
        pyyaml \
        scipy==1.11.4 \
        numpy==1.26.4 \
        pandas==2.0.3 \
        shapely==2.0.7 \
        tensorboard==2.14.0 \
        pytest==7.4.0 \
        portalocker==2.4.0

COPY . .

RUN pip install -e . --no-deps

ENV PYTHONPATH=/workspace

CMD ["bash"]
