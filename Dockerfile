# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=ubuntu:22.04
ARG BASE_RUNTIME_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

FROM ${BASE_IMAGE} AS python-env

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ARG PIP_NO_CACHE_DIR=1
ARG PYENV_VERSION=v2.3.27
ARG PYTHON_VERSION=3.10.13

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


FROM ${BASE_RUNTIME_IMAGE} AS runtime-env

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ARG PIP_NO_CACHE_DIR=1
ENV PATH=/home/user/.local/bin:/opt/python/bin:${PATH}

COPY --from=python-env /opt/python /opt/python

RUN <<EOF
    set -eu

    apt-get update
    apt-get install -y \
        git \
        tk \
        libglib2.0-0 \
        gosu
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
    set -eu

    groupadd -o -g 1000 user
    useradd -m -o -u 1000 -g user user
EOF

ARG SD_SCRIPTS_URL=https://github.com/kohya-ss/sd-scripts
# v0.6.6
ARG SD_SCRIPTS_VERSION=54500b861dff5bc1c6b555733d355a320935bf34

RUN <<EOF
    set -eu

    mkdir -p /code
    chown -R user:user /code

    gosu user git clone "${SD_SCRIPTS_URL}" /code/sd-scripts
    cd /code/sd-scripts
    gosu user git checkout "${SD_SCRIPTS_VERSION}"
    gosu user git submodule update --init
EOF

WORKDIR /code/sd-scripts
ADD ./requirements.txt /code/
RUN <<EOF
    set -eu

    cd /code/
    gosu user pip3 install --no-cache-dir -r ./requirements.txt

    cd /code/sd-scripts/
    gosu user pip3 install --no-cache-dir .
EOF

RUN <<EOF
    set -eu

    # gosu user accelerate config
    gosu user mkdir -p /home/user/.cache/huggingface/accelerate
    gosu user tee /home/user/.cache/huggingface/accelerate/default_config.yaml <<EOT
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

EOF

ENTRYPOINT [ "gosu", "user", "accelerate", "launch" ]
