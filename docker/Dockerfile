FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ARG DOCKER_UID=1000
ARG DOCKER_USER=ubuntu
ARG DOCKER_PASSWORD=ubuntu

USER root
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN set -x && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    apt update && apt upgrade -y && apt autoremove && apt clean && \
    apt install -y tzdata git cmake build-essential subversion sudo openssh-server python3-pip && \
    echo root:root | chpasswd && \
    chsh -s /bin/bash && \
    useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

USER ${DOCKER_USER}
ENV PATH $PATH:/home/${DOCKER_USER}/.local/bin
RUN set -x && \
    python3 -m pip install torch==2.0.1 transformers==4.31.0 peft==0.4.0 numpy==1.23.5 torchinfo==1.8.0 tqdm==4.64.1