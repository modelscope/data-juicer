# The data-juicer image includes all open-source contents of data-juicer,
# and it will be instaled in editable mode.

FROM python:3.8.18

# prepare the java env
WORKDIR /opt
# download jdk
RUN wget https://aka.ms/download-jdk/microsoft-jdk-17.0.9-linux-x64.tar.gz -O jdk.tar.gz && \
    tar -xzf jdk.tar.gz && \
    rm -rf jdk.tar.gz && \
    mv jdk-17.0.9+8 jdk

# set the environment variable
ENV JAVA_HOME=/opt/jdk

WORKDIR /data-juicer

# install requirements first to better reuse installed library cache
COPY environments/ environments/
RUN cat environments/* | xargs pip install --default-timeout 1000

# install data-juicer then
COPY . .
RUN pip install -v -e .[all]
