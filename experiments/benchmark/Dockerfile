FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git swig libegl1 libgl1 libgles2 libosmesa6 ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
ENV UV_HTTP_TIMEOUT=120
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /opt/conda/bin/python torch==2.6.0 torchvision==0.21.0 stable-worldmodel==0.1.1 stable-pretraining==0.1.7 'transformers<5' h5py hdf5plugin scipy pytest pygame pymunk shapely imageio imageio-ffmpeg
WORKDIR /workspace
ENV PYTHONPATH=/workspace/src STABLEWM_HOME=/data
