# syntax=docker/dockerfile:1
# Docker file
FROM python:slim
WORKDIR /usr/src/app

# Runtime dependencies:
RUN set -eux; \
    apt-get update;\
    apt-get install -y --no-install-recommends \
              libproj-dev libeigen3-dev \
              libcgal-dev libgeographic-dev \
              build-essential python3-sphinx ninja-build git \
              libopenblas-dev libopenblas-base liblapacke-dev libgsl-dev \
              python3-numpy python3-scipy

# CMake toolchain:
RUN set -eux; \
    apt-get install -y --no-install-recommends cmake cmake-data cmake-extras

# Install Cython:
RUN set -eux; \
    pip install Cython

# Python dependencies:
RUN set -eux; \
    pip install matplotlib pyproj meson; \
    pip install 'mebuex @ git+https://github.com/mjziebarth/Mebuex';
RUN set -eux; \
    pip install 'loaducerf3 @ git+https://git.gfz-potsdam.de/ziebarth/loaducerf3';

# Copy necessary directories:
COPY ./include/ ./include/
COPY ./docs/ ./docs/
COPY ./external/ ./external/
COPY ./jupyter/ ./jupyter/
COPY ./reheatfunq/ ./reheatfunq/
COPY ./src/ ./src/
COPY ./compile.sh ./meson.build ./setup.py ./

# Fix GeographicLib CMake error on the docker image:
RUN ln -s /usr/share/cmake/geographiclib/FindGeographicLib.cmake \
         /usr/share/cmake-3.18/Modules/

# Compile and install the package:
RUN set -eux; \
    bash compile.sh;
RUN set -eux; \
    pip install --user .

# Install the optional dependencies for the Jupyter notebooks
RUN set -eux; \
    apt-get install -y --no-install-recommends jupyter-notebook
RUN set -eux; \
    pip install cmcrameri cmocean shapely cmasher sklearn joblib geopandas
RUN set -eux; \
    pip install 'flottekarte @ git+https://github.com/mjziebarth/FlotteKarte.git'; \
    pip install 'pdtoolbox @ git+https://git.gfz-potsdam.de/ziebarth/pdtoolbox.git';