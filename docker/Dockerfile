FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DOCKER_UID=1000
ARG DOCKER_USER=ubuntu
ARG DOCKER_PASSWORD=ubuntu

USER root
RUN set -x && \
    apt update && apt upgrade -y && apt autoremove && apt clean && \
    apt install -y git cmake build-essential aptitude graphviz subversion sudo openssh-server && \
    echo root:root | chpasswd && \
    chsh -s /bin/bash && \
    useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

USER ${DOCKER_USER}
ENV PATH $PATH:/home/${DOCKER_USER}/.local/bin
RUN set -x && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install transformers==4.31.0