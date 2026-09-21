FROM ghcr.io/darkstar1997/opencv-cuda:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    ninja-build \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cuda-python==12.8
RUN pip install --no-cache-dir torch==2.8 torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir "numpy<2" scipy==1.13.1 scikit-learn==1.4.2
RUN pip install --no-cache-dir absl-py attrs flatbuffers "protobuf<5,>=4.25.3" matplotlib
RUN pip install --no-cache-dir "setuptools<82" wheel packaging
RUN pip install --no-cache-dir tqdm==4.66.4
RUN pip install --no-cache-dir pandas==2.2.2
RUN pip install --no-cache-dir Pillow==10.3.0
RUN pip install --no-cache-dir seaborn==0.13.2
RUN pip install --no-cache-dir --no-deps grad-cam ttach
RUN pip install --no-cache-dir "numpy<2" scikit-image==0.24.0 openslide-python
RUN pip install --no-cache-dir timm==1.0.15 einops==0.8.1 yacs==0.1.8 fvcore
# Mamba-UNet runtime.  Keep this in the image: launchers use disposable
# containers and must never install binary dependencies at experiment runtime.
RUN pip install --no-cache-dir transformers==4.56.2
RUN pip install --no-cache-dir pytest==8.3.5
COPY scripts/install_mamba_ssm.py /tmp/install_mamba_ssm.py
RUN python3 /tmp/install_mamba_ssm.py
# Runtime dependencies used by the pinned official HoVer-Net trainer.
RUN pip install --no-cache-dir docopt==0.6.2 tensorboardX==2.6.4 termcolor==2.4.0 future==1.0.0 shapely \
    && pip install --no-cache-dir --no-deps imgaug==0.4.0

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN mkdir -p /app/data /app/src /app/outputs/models /app/outputs/results

CMD ["bash", "/app/runScript.sh"]
