# syntax=docker/dockerfile:1.12
ARG BASE_IMAGE=ubuntu:22.04
ARG BASE_RUNTIME_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

FROM ${BASE_IMAGE} AS build-python-stage

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ARG PIP_NO_CACHE_DIR=1
ARG PYENV_VERSION=v2.5.0
ARG PYTHON_VERSION=3.10.16

RUN <<EOF
    set -eu

    apt-get update

    apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        curl \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        git

    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
    set -eu

    git clone https://github.com/pyenv/pyenv.git /opt/pyenv
    cd /opt/pyenv
    git checkout "${PYENV_VERSION}"

    PREFIX=/opt/python-build /opt/pyenv/plugins/python-build/install.sh
    /opt/python-build/bin/python-build -v "${PYTHON_VERSION}" /opt/python

    rm -rf /opt/python-build /opt/pyenv
EOF

FROM ${BASE_RUNTIME_IMAGE} AS build-python-venv-stage

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN <<EOF
    set -eu

    apt-get update
    apt-get install -y \
        git \
        gosu

    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

ARG VENV_BUILDER_UID=999
ARG VENV_BUILDER_GID=999
RUN <<EOF
    set -eu

    groupadd --non-unique --gid "${VENV_BUILDER_GID}" venvbuilder
    useradd --non-unique --uid "${VENV_BUILDER_UID}" --gid "${VENV_BUILDER_GID}" --create-home venvbuilder
EOF

COPY --from=build-python-stage --chown=root:root /opt/python /opt/python
ENV PATH="/opt/python/bin:${PATH}"

RUN <<EOF
    set -eu

    mkdir -p /opt/python_venv
    chown -R "${VENV_BUILDER_UID}:${VENV_BUILDER_GID}" /opt/python_venv

    gosu venvbuilder python -m venv /opt/python_venv
EOF
ENV PATH="/opt/python_venv/bin:${PATH}"

COPY --chown=root:root ./requirements.txt /python_venv_tmp/
RUN --mount=type=cache,uid=${VENV_BUILDER_UID},gid=${VENV_BUILDER_GID},target=/home/venvbuilder/.cache/pip <<EOF
    set -eu

    gosu venvbuilder pip install -r /python_venv_tmp/requirements.txt
EOF

ARG SD_SCRIPTS_URL=https://github.com/kohya-ss/sd-scripts
# v0.7.0
ARG SD_SCRIPTS_VERSION=2a23713f71628b2d1b88a51035b3e4ee2b5dbe46

RUN <<EOF
    set -eu

    mkdir -p /opt/sd-scripts
    chown -R "${VENV_BUILDER_UID}:${VENV_BUILDER_GID}" /opt/sd-scripts

    gosu venvbuilder git clone "${SD_SCRIPTS_URL}" /opt/sd-scripts
    cd /opt/sd-scripts
    gosu venvbuilder git checkout "${SD_SCRIPTS_VERSION}"
    gosu venvbuilder git submodule update --init
EOF

RUN --mount=type=cache,uid=${VENV_BUILDER_UID},gid=${VENV_BUILDER_GID},target=/home/venvbuilder/.cache/pip <<EOF
    set -eu

    cd /opt/sd-scripts/
    gosu venvbuilder python -m compileall .
    gosu venvbuilder pip install --no-deps --editable .
EOF


FROM ${BASE_RUNTIME_IMAGE} AS runtime-env

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN <<EOF
    set -eu

    apt-get update
    apt-get install -y \
        git \
        tk \
        libglib2.0-0

    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# libnvrtc.so workaround
# https://github.com/aoirint/sd-scripts-docker/issues/19
RUN <<EOF
    set -eu

    ln -s \
        /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so.11.2 \
        /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvrtc.so
EOF

COPY --from=build-python-stage --chown=root:root /opt/python /opt/python
COPY --from=build-python-venv-stage --chown=root:root /opt/python_venv /opt/python_venv
COPY --from=build-python-venv-stage --chown=root:root /opt/sd-scripts /opt/sd-scripts
ENV PATH="/opt/python_venv/bin:${PATH}"

WORKDIR /opt/sd-scripts

# huggingface cache dir
ENV HF_HOME=/huggingface

RUN <<EOF
    set -eu

    # create huggingface cache dir
    mkdir -p /huggingface

    # create accelerate cache dir
    mkdir -p /huggingface/accelerate

    tee /huggingface/accelerate/default_config.yaml <<EOT
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
EOT

    # writable by default execution user
    chown -R "1000:1000" /huggingface
EOF

USER "1000:1000"
ENTRYPOINT [ "accelerate", "launch" ]
