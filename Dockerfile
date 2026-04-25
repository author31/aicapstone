# syntax = docker/dockerfile:1.7
ARG CUDA_VERSION=12.2.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS uv-base

COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /usr/bin/

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

COPY --from=uv-base /usr/bin/uv /usr/bin/uv
COPY --from=uv-base /usr/bin/uvx /usr/bin/uvx

WORKDIR /workspace/aicapstone

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/workspace/aicapstone/.venv \
    PYTHONPATH=/workspace/aicapstone/packages/umi/src:/workspace/aicapstone/packages/leisaac/source/leisaac

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git cmake build-essential curl wget gnupg2 lsb-release \
        software-properties-common locales pkg-config ca-certificates \
        python3.11 python3.11-dev python3.11-distutils \
        libboost-all-dev libqhull-dev libassimp-dev liboctomap-dev \
        libconsole-bridge-dev libfcl-dev libeigen3-dev \
        libx11-dev libxaw7-dev libxrandr-dev libgl1-mesa-dev libglu1-mesa-dev \
        libglew-dev libgles2-mesa-dev libopengl-dev libfreetype-dev \
        qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
        libyaml-cpp-dev libzzip-dev freeglut3-dev libogre-1.9-dev \
        libpng-dev libjpeg-dev python3-pyqt5.qtwebengine \
        libbullet-dev libasio-dev libtinyxml2-dev \
        libcunit1-dev libacl1-dev libfmt-dev \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100

COPY . /workspace/aicapstone

RUN uv venv --python /usr/bin/python3.11 ${VIRTUAL_ENV} && \
    echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python ${VIRTUAL_ENV}

RUN ln -sf /usr/include/python3.11 /usr/include/python3

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
