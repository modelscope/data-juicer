# The data-juicer image includes all open-source contents of data-juicer,
# and it will be instaled in editable mode.

FROM python:3.8.18

WORKDIR /data-juicer

# install requirements first to better reuse installed library cache
COPY environments/ environments/
RUN cat environments/* | xargs pip install

# install data-juicer then
COPY . .
RUN pip install -v -e .[all]
