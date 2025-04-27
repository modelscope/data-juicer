# The data-juicer image includes all open-source contents of data-juicer,
# and it will be installed in editable mode.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# change to aliyun source
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# install python 3.10
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y git curl vim wget python3.10 libpython3.10-dev python3-pip libgl1-mesa-glx libglib2.0-0 \
    && ln -sf /usr/bin/python3.10  /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10  /usr/bin/python \
    && apt-get autoclean && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# install 3rd-party system dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 software-properties-common build-essential cmake gfortran libopenblas-dev liblapack-dev postgresql postgresql-contrib libpq-dev

# prepare the java env
WORKDIR /opt
# download jdk
RUN wget https://aka.ms/download-jdk/microsoft-jdk-17.0.9-linux-x64.tar.gz -O jdk.tar.gz \
    && tar -xzf jdk.tar.gz \
    && rm -rf jdk.tar.gz \
    && mv jdk-17.0.9+8 jdk

# set the environment variable
ENV JAVA_HOME=/opt/jdk

WORKDIR /data-juicer

# install requirements which need to be installed from source
RUN pip install --upgrade setuptools==69.5.1 setuptools_scm \
    && pip install git+https://github.com/xinyu1205/recognize-anything.git --default-timeout 1000

# install data-juicer then
COPY . .
RUN pip install -v -e .[all] --default-timeout 1000 \
    && python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger');  nltk.download('averaged_perceptron_tagger_eng')"
