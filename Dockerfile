# syntax=docker/dockerfile:1
# Docker file
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
#               Jupyter Development Team (see below)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
              python3-numpy cmake; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Install Cython:
RUN set -eux; \
    pip install --no-cache-dir Cython meson

# Add the user that will later be the default:
RUN set -eux;\
    useradd -m -d /home/reheatfunq -p 12345 reheatfunq
USER reheatfunq
RUN set -eux;\
    mkdir /home/reheatfunq/REHEATFUNQ
WORKDIR /home/reheatfunq/REHEATFUNQ

# Python dependencies:
RUN set -eux; \
    pip install --no-cache-dir --user \
             matplotlib pyproj; \
    pip install --no-cache-dir --user \
            'mebuex @ git+https://github.com/mjziebarth/Mebuex';
RUN set -eux; \
    pip install --no-cache-dir --user \
            'loaducerf3 @ git+https://git.gfz-potsdam.de/ziebarth/loaducerf3';

# Install the optional dependencies for the Jupyter notebooks
RUN set -eux; \
    pip install --user --no-cache-dir \
            notebook cmcrameri cmocean shapely \
            cmasher scikit-learn joblib geopandas scipy requests
# FlotteKarte currently needs a hotfix to install:
RUN set -eux; \
    git clone https://github.com/mjziebarth/FlotteKarte.git; \
    cd FlotteKarte; \
    bash compile.sh; \
    pip install --user --no-cache-dir .
RUN set -eux; \
    pip install --user  --no-cache-dir \
            'pdtoolbox @ git+https://git.gfz-potsdam.de/ziebarth/pdtoolbox.git';
ENV PATH = "$PATH:/home/reheatfunq/.local/bin"

# Copy necessary directories:
COPY ./include/ ./include/
COPY ./external/ ./external/
COPY ./reheatfunq/ ./reheatfunq/
COPY ./src/ ./src/
COPY ./compile.sh ./meson.build ./setup.py ./

# Compile and install the package:
RUN set -eux; \
    bash compile.sh; \
    pip install --no-cache-dir --user .; \
    rm -r build; \
    rm -r builddir


RUN set -eux; \
    jupyter notebook --generate-config; \
    mkdir -p /home/reheatfunq/jupyter/REHEATFUNQ/data
WORKDIR /home/reheatfunq
COPY ./jupyter/ jupyter/
WORKDIR /home/reheatfunq/jupyter

# Set permissions to reheatfunq user:
USER root
RUN set -eux; \
    chown -R reheatfunq /home/reheatfunq/
USER reheatfunq

# Define the command that the image will execute:
CMD jupyter notebook --no-browser --ip 0.0.0.0 --port=8888 --debug -y
EXPOSE 8888

# The following HEALTHCHECK block is from docker-stacks/base-notebook/Dockerfile
#
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
#
# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK  --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -O- --no-verbose --tries=1 --no-check-certificate \
    http${GEN_CERT:+s}://localhost:8888${JUPYTERHUB_SERVICE_PREFIX:-/}api || exit 1