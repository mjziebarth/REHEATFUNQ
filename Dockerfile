# syntax=docker/dockerfile:1
# Docker file
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022-2023 Malte J. Ziebarth
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

#
# This first part of the image creates a basic running linux
# with a gcc build infrastructure. Should the creation of this
# image ever fail in the future, the following parts of the code
# have to be adjusted to create a minimal linux with build
# facilities.
#
FROM debian:bookworm-20221219-slim

USER root

# First installs:
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
                    gcc g++ libc-dev netbase;\
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

#
# Hereafter, everything should be self-contained.
#

ENV PREFIX=/sci

# Add the user that will later be the default:
RUN set -eux;\
    useradd -m -d /home/reheatfunq -p 12345 reheatfunq
USER reheatfunq
RUN set -eux;\
    mkdir /home/reheatfunq/REHEATFUNQ
WORKDIR /home/reheatfunq/REHEATFUNQ

USER root

# Bootstrap GCC:
COPY ./vendor/xz-5.4.0.tar.bz2 \
     ./vendor/make-4.4.tar.gz \
     ./vendor/gcc-12.2.0.tar.xz \
     ./vendor/bzip2-1.0.8.tar.gz \
     ./vendor/mpfr-4.2.0.tar.xz \
     ./vendor/mpc-1.3.1.tar.xz \
     ./vendor/gmp-6.2.1.tar.xz \
     ./vendor/m4-1.4.19.tar.xz \
     ./vendor/perl-5.36.0.tar.xz \
     ./vendor/

COPY ./docker/bootstrap-gcc.sh ./docker/

RUN set -eux; \
    PREFIX=$PREFIX \
    VENDORDIR=vendor \
    BZ2_ID=bzip2-1.0.8 \
    XZ_ID=xz-5.4.0 \
    M4_ID=m4-1.4.19 \
    GMP_ID=gmp-6.2.1 \
    MPFR_ID=mpfr-4.2.0 \
    MPC_ID=mpc-1.3.1 \
    GCC_ID=gcc-12.2.0 \
    MAKE_ID=make-4.4 \
    PERL_ID=perl-5.36.0 \
    ./docker/bootstrap-gcc.sh

RUN set -eux; \
    apt-get remove -y --no-install-recommends \
                    gcc g++ libc-dev;\
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

#RUN set -eux; \
#    echo "/sci/lib" > /etc/ld.so.conf; \
#    echo "/sci/lib64" >> /etc/ld.so.conf; \
#    echo "/usr/local/lib" >> /etc/ld.so.conf; \
#    echo "/usr/local/lib64" >> /etc/ld.so.conf; \
#    cat /etc/ld.so.conf.old >> /etc/ld.so.conf; \
#    ldconfig; \
#    exit -1
RUN set -eux; \
    cd /sci/bin; \
    ln -s gcc cc; \
    ls -l /sci/bin


# Runtime dependencies:
# Copy external code that includes dependencies to install
COPY ./vendor/sqlite-amalgamation-3400100.tar.xz \
     ./vendor/eigen-3.4.0.tar.xz ./vendor/boost_1_81_0.tar.lzma-lrz \
     ./vendor/proj-9.1.1.tar.xz ./vendor/geos-3.11.1.tar.xz \
     ./vendor/lrzip-0.651.tar.gz  ./vendor/autoconf-2.71.tar.xz \
     ./vendor/automake-1.16.5.tar.xz \
     ./vendor/libtool-2.4.6.tar.xz ./vendor/zlib-1.2.13.tar.xz \
     ./vendor/lzo-2.10.tar.gz ./vendor/lz4-1.9.4.tar.gz \
     ./vendor/libzmq-master-2023-01-10.tar.xz \
     ./vendor/openssl-3.0.7.tar.xz \
     ./vendor/cmake-3.25.1.tar.xz ./vendor/ninja-1.11.1.tar.xz \
     ./vendor/libffi-3.4.4.tar.xz \
     ./vendor/geos-3.11.1.tar.xz ./vendor/Python-3.11.1.tar.xz \
     ./vendor/Cython-0.29.33.tar.xz ./vendor/OpenBLAS-0.3.21.tar.xz \
     ./vendor/pandas-1.5.2.tar.xz ./vendor/shapely-2.0.0.tar.xz \
     ./vendor/Fiona-1.8.22.tar.xz ./vendor/cffi-1.15.1.tar.xz  \
     ./vendor/argon2-cffi-bindings-21.2.0.tar.xz \
     ./vendor/MarkupSafe-2.1.1.tar.xz ./vendor/pyzmq-24.0.1.tar.xz \
     ./vendor/tornado-6.2.tar.xz ./vendor/pyrsistent-0.19.3.tar.gz \
     ./vendor/PyYAML-6.0.tar.gz ./vendor/debugpy-1.6.5.tar.xz \
     ./vendor/psutil-5.9.4.tar.xz ./vendor/scikit-learn-1.2.0.tar.xz \
     ./vendor/numpy-1.24.1.tar.xz ./vendor/scipy-1.10.0.tar.xz \
     ./vendor/Pillow-9.4.0.tar.lzma-lrz ./vendor/contourpy-1.0.6.tar.xz \
     ./vendor/kiwisolver-1.4.4.tar.xz ./vendor/matplotlib-3.6.2.tar.xz \
     ./vendor/pyproj-3.4.1.tar.xz ./vendor/gdal-3.6.2.tar.xz \
     ./vendor/yaml-0.2.5.tar.xz ./vendor/CGAL-5.5.1-library.tar.xz \
     ./vendor/gsl-2.7.1.tar.xz \
     ./vendor/geographiclib-2.1.2.tar.xz \
     ./vendor/
COPY ./include/ ./include/
COPY ./vendor/patch/ ./vendor/patch/

# Install bz2, xz-utils, m4, autoconf, automake, libtool, lrzip, perl, OpenSSL,
#         cmake, Ninja:
RUN set -eux; \
    export PATH=$PREFIX/bin:$PATH; \
    export CPATH=$PREFIX/include; \
    ls -la /usr/bin; \
    which make; \
    which gcc; \
    which cc; \
    #
    # m4:
    #
    tar -xf vendor/m4-1.4.19.tar.xz m4-1.4.19; \
    cd m4-1.4.19; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf m4-1.4.19; \
    #
    # autoconf:
    #
    tar -xf vendor/autoconf-2.71.tar.xz autoconf-2.71; \
    cd autoconf-2.71/; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf autoconf-2.71; \
    #
    # automake:
    #
    tar -xf vendor/automake-1.16.5.tar.xz automake-1.16.5; \
    cd automake-1.16.5/; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf automake-1.16.5; \
    #
    # libtool:
    #
    tar -xf vendor/libtool-2.4.6.tar.xz libtool-2.4.6; \
    cd libtool-2.4.6/; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf libtool-2.4.6; \
    #
    # zlib:
    #
    tar -xf vendor/zlib-1.2.13.tar.xz zlib-1.2.13; \
    cd zlib-1.2.13/; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf zlib-1.2.13; \
    #
    # lzo:
    #
    tar -xf vendor/lzo-2.10.tar.gz lzo-2.10; \
    cd lzo-2.10/; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf lzo-2.10; \
    #
    # lz4:
    #
    tar -xf vendor/lz4-1.9.4.tar.gz lz4-1.9.4; \
    cd lz4-1.9.4/; \
    make -j `nproc`; \
    make install PREFIX=$PREFIX; \
    cd ..; \
    rm -rf lz4-1.9.4; \
    #
    # lrzip:
    #
    tar -xf vendor/lrzip-0.651.tar.gz lrzip-0.651; \
    cd lrzip-0.651/; \
    ./autogen.sh; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf lrzip-0.651; \
    #
    # OpenSSL:
    #
    tar -xf vendor/openssl-3.0.7.tar.xz; \
    cd openssl-3.0.7; \
    ./Configure --prefix=$PREFIX zlib; \
    make -j `nproc`; \
    make install_sw; \
    cd ..; \
    rm -rf openssl-3.0.7; \
    #
    # cmake:
    #
    tar -xf vendor/cmake-3.25.1.tar.xz; \
    cd cmake-3.25.1; \
    ./bootstrap --parallel=`nproc` -- \
                -DCMAKE_INSTALL_PREFIX=$PREFIX \
                -DCMAKE_BUILD_TYPE:STRING=Release; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf cmake-3.25.1; \
    #
    # Ninja:
    #
    tar -xf vendor/ninja-1.11.1.tar.xz ninja-1.11.1; \
    cd ninja-1.11.1; \
    cmake -Bbuild-cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=$PREFIX; \
    cmake --build build-cmake -j `nproc`; \
    ./build-cmake/ninja_test; \
    cp build-cmake/ninja /bin; \
    cd ..; \
    rm -rf ninja-1.11.1 ; \
    #
    # libzmq:
    #
    tar -xf vendor/libzmq-master-2023-01-10.tar.xz libzmq-master-2023-01-10; \
    cd libzmq-master-2023-01-10/; \
    mkdir cmake-ninja; \
    cd cmake-ninja; \
    cmake -G Ninja -D CMAKE_BUILD_TYPE=Release \
                   -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_LIB_DIR=lib ..; \
    cmake --build .; \
    cmake --build . --target install; \
    cd ../..; \
    rm -rf libzmq-master-2023-01-10

ENV PATH "/sci/bin:$PATH"
ENV CPATH="/sci/include/"

# Install sqlite3:
COPY ./external/sqlite ./external/sqlite
RUN set -eux; \
    tar -xf vendor/sqlite-amalgamation-3400100.tar.xz \
            sqlite-amalgamation-3400100; \
    mv sqlite-amalgamation-3400100/* external/sqlite; \
    rmdir sqlite-amalgamation-3400100; \
    cd external/sqlite/; \
    mkdir build; \
    cd build; \
    cmake -G Ninja -D CMAKE_BUILD_TYPE=Release \
                   -D CMAKE_INSTALL_PREFIX=$PREFIX ..; \
    cmake --build .; \
    cmake --build . --target install; \
    ls -l /usr/lib; \
    cd ../../..; \
    # Ensure that the RTREE works: \
    gcc vendor/patch/sqlite3/ensure_rtree.cpp -lsqlite3 -o ensure_rtree; \
    rm ensure_rtree; \
    rm -rf external/sqlite

# Install Proj:
RUN set -eux; \
    #export CMAKE_PROGRAM_PATH=/usr/bin; \
    tar -xf vendor/proj-9.1.1.tar.xz proj-9.1.1; \
    pwd; \
    cd proj-9.1.1;\
    mkdir build; \
    cd build; \
    cmake -G Ninja -DENABLE_TIFF=OFF -DENABLE_CURL=OFF -DBUILD_PROJSYNC=OFF \
          -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_INSTALL_LIBDIR=lib ..; \
    cmake --build .; \
    cmake --build . --target install; \
    cd ..; \
    rm -rf proj-9.1.1/;

# Install boost:
RUN set -eux; \
    lrunzip vendor/boost_1_81_0.tar.lzma-lrz -o ./boost_1_81_0.tar; \
    tar xvf boost_1_81_0.tar; \
    rm -vf boost_1_81_0.tar; \
    cd boost_1_81_0/; \
    ./bootstrap.sh --with-libraries=math --prefix=$PREFIX;\
    ./b2 install; \
    cd /home/reheatfunq/REHEATFUNQ; \
    rm -rf boost_1_81_0/; \
    #
    # Eigen:
    #
    tar -xf vendor/eigen-3.4.0.tar.xz eigen-3.4.0/Eigen; \
    mkdir include/eigen3/; \
    mv eigen-3.4.0/Eigen/ include/eigen3/Eigen; \
    rm -rf eigen-3.4.0/; \
    #
    # libffi:
    #
    tar -xf vendor/libffi-3.4.4.tar.xz libffi-3.4.4; \
    cd libffi-3.4.4; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf libffi-3.4.4

# Install Python:
RUN set -eux; \
    #
    # Python:
    #
    tar -xf vendor/Python-3.11.1.tar.xz Python-3.11.1; \
    cd Python-3.11.1; \
    env TESTTIMEOUT=10; \
    export PYLDFLAGSONE="-L$PREFIX/lib -L$PREFIX/lib64 "; \
    export PYLDFLAGSTWO="-Wl,--rpath=$PREFIX/lib -Wl,--rpath=$PREFIX/lib64"; \
    ./configure --enable-optimizations --with-lto=yes --enable-shared \
                --with-assertions --prefix=$PREFIX --with-openssl=/sci \
                CFLAGS="-I/sci/include/" \
                LDFLAGS="$PYLDFLAGSONE $PYLDFLAGSTWO"; \
    make -j `nproc`; \
    #make test; \
    make install; \
    cd ..; \
    rm -rf Python-3.11.1; \
    ln -s $PREFIX/bin/pip3 $PREFIX/bin/pip; \
    ln -s $PREFIX/bin/python3 $PREFIX/bin/python; \
    find /sci -name libpython*; \
    python vendor/patch/python/test_bz2.py

COPY ./vendor/wheels/ ./vendor/wheels/
COPY ./vendor/geos-3.11.1.tar.xz \
     ./vendor/Cython-0.29.33.tar.xz \
     ./vendor/OpenBLAS-0.3.21.tar.xz \
     ./vendor/numpy-1.24.1.tar.xz \
     ./vendor/geographiclib-2.1.2.tar.xz \
     ./vendor/gsl-2.7.1.tar.xz \
     ./vendor/gdal-3.6.2.tar.xz \
     ./vendor/yaml-0.2.5.tar.xz \
     ./vendor/gmp-6.2.1.tar.xz \
     ./vendor/CGAL-5.5.1-library.tar.xz \
     ./vendor/patch-2.7.6.tar.xz \
     ./vendor/

RUN set -eux; \
    #
    # patch:
    #
    tar -xf vendor/patch-2.7.6.tar.xz patch-2.7.6; \
    cd patch-2.7.6; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf patch-2.7.6; \
    #
    # GEOS:
    #
    tar -xf vendor/geos-3.11.1.tar.xz geos-3.11.1; \
    cd geos-3.11.1; \
    mkdir build; \
    cd build;\
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release \
                   -DCMAKE_INSTALL_LIBDIR=lib ..; \
    cmake --build .; \
    cmake --build . --target install; \
    cd ../..; \
    rm -rf geos-3.11.1; \
    #
    # wheel:
    #
    pip install --no-index --no-build-isolation\
         vendor/wheels/wheel-0.38.4-py3-none-any.whl; \
    #         \
    # Cython: \
    #         \
    tar -xf vendor/Cython-0.29.33.tar.xz Cython-0.29.33; \
    cd Cython-0.29.33; \
    #python setup.py install -j`nproc`; \
    python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --no-index --no-build-isolation \
                dist/Cython-0.29.33-*.whl; \
    cd ..; \
    rm -rf Cython-0.29.33; \
    pip install --no-cache-dir vendor/wheels/meson-1.0.0-py3-none-any.whl; \
    #           \
    # OpenBLAS: \
    #           \
    tar -xf vendor/OpenBLAS-0.3.21.tar.xz; \
    cd OpenBLAS-0.3.21; \
    make -j `nproc`; \
    make install PREFIX=$PREFIX; \
    cd ..; \
    rm -rf OpenBLAS-0.3.21; \
    #
    # Numpy:
    #
    tar -xf vendor/numpy-1.24.1.tar.xz numpy-1.24.1; \
    cp vendor/patch/NumPy/site.cfg numpy-1.24.1/; \
    cd numpy-1.24.1; \
    python setup.py build -j `nproc` install --prefix $PREFIX; \
    cd ..; \
    rm -rf numpy-1.24.1; \
    #
    # GeographicLib:
    #
    tar -xf vendor/geographiclib-2.1.2.tar.xz geographiclib-2.1.2/; \
    cd geographiclib-2.1.2; \
    mkdir build; \
    cd build; \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$PREFIX \
                   -DCMAKE_BUILD_TYPE=Release ..; \
    cmake --build .; \
    cmake --build . --target install; \
    cd ../..; \
    rm -rf geographiclib-2.1.2; \
    #
    # GSL:
    #
    tar -xf vendor/gsl-2.7.1.tar.xz gsl-2.7.1; \
    cd gsl-2.7.1; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf gsl-2.7.1; \
    #
    # GDAL:
    #
    tar -xf vendor/gdal-3.6.2.tar.xz; \
    cd gdal-3.6.2; \
    mkdir build; \
    cd build; \
    cmake -G Ninja  -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release \
                    -DPython_ROOT=$PREFIX ..; \
    cmake --build .; \
    cmake --build . --target install; \
    cd ../..; \
    rm -rf gdal-3.6.2; \
    #
    # YAML:
    #
    tar -xf vendor/yaml-0.2.5.tar.xz yaml-0.2.5; \
    cd yaml-0.2.5; \
    ./configure --prefix=$PREFIX; \
    make -j `nproc`; \
    make install; \
    cd ..; \
    rm -rf yaml-0.2.5; \
    #              \
    # CGAL:  \
    #              \
    tar -xf vendor/CGAL-5.5.1-library.tar.xz CGAL-5.5.1; \
    cd CGAL-5.5.1; \
    cmake . -D CMAKE_INSTALL_PREFIX=$PREFIX -D CMAKE_BUILD_TYPE=Release; \
    make install; \
    cd ..; \
    rm -rf CGAL-5.5.1;

# Pythran:

RUN set -eux; \
    #          \
    # Pythran, pybind11, ply, beniget, gast: \
    #          \
    pip install --no-index --no-build-isolation\
                vendor/wheels/pythran-0.12.0-py3-none-any.whl \
                vendor/wheels/pybind11-2.10.3-py3-none-any.whl \
                vendor/wheels/gast-0.5.3-py3-none-any.whl \
                vendor/wheels/beniget-0.4.1-py3-none-any.whl \
                vendor/wheels/cycler-0.11.0-py3-none-any.whl \
                vendor/wheels/packaging-23.0-py3-none-any.whl \
                vendor/wheels/certifi-2022.12.7-py3-none-any.whl \
                vendor/wheels/six-1.16.0-py2.py3-none-any.whl \
                vendor/wheels/setuptools-65.6.3-py3-none-any.whl \
                vendor/wheels/cppy-1.2.1-py3-none-any.whl \
                vendor/wheels/ply-3.11-py2.py3-none-any.whl

USER reheatfunq

COPY ./vendor/Pillow-9.4.0.tar.lzma-lrz \
     ./vendor/contourpy-1.0.6.tar.xz \
     ./vendor/kiwisolver-1.4.4.tar.xz \
     ./vendor/matplotlib-3.6.2.tar.xz \
     ./vendor/pyproj-3.4.1.tar.xz \
     ./vendor/scipy-1.10.0.tar.xz \
     ./vendor/pandas-1.5.2.tar.xz \
     ./vendor/shapely-2.0.0.tar.xz \
     ./vendor/Fiona-1.8.22.tar.xz \
     ./vendor/cffi-1.15.1.tar.xz \
     ./vendor/argon2-cffi-bindings-21.2.0.tar.xz \
     ./vendor/MarkupSafe-2.1.1.tar.xz \
     ./vendor/pyzmq-24.0.1.tar.xz \
     ./vendor/tornado-6.2.tar.xz \
     ./vendor/pyrsistent-0.19.3.tar.gz \
     ./vendor/PyYAML-6.0.tar.gz \
     ./vendor/debugpy-1.6.5.tar.xz \
     ./vendor/psutil-5.9.4.tar.xz \
     ./vendor/scikit-learn-1.2.0.tar.xz \
     ./vendor/

RUN set -eux; \
    #         \
    # Pillow: \
    #         \
    lrunzip vendor/Pillow-9.4.0.tar.lzma-lrz -o ./Pillow-9.4.0.tar; \
    tar xvf Pillow-9.4.0.tar; \
    rm -vf Pillow-9.4.0.tar; \
    cd Pillow-9.4.0; \
    CFLAGS="-flto=`nproc`" python setup.py build_ext -j`nproc` \
        --disable-jpeg bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/Pillow-9.4.0-*.whl; \
    cd ..; \
    rm -rf Pillow-9.4.0; \
    #            \
    # Contourpy: \
    #            \
    tar -xf vendor/contourpy-1.0.6.tar.xz contourpy-1.0.6; \
    cd contourpy-1.0.6; \
    CFLAGS="-flto=`nproc`" python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/contourpy-1.0.6-*.whl; \
    cd ..; \
    rm -rf contourpy-1.0.6; \
    #             \
    # Kiwisolver: \
    #             \
    tar -xf vendor/kiwisolver-1.4.4.tar.xz kiwisolver-1.4.4; \
    cd kiwisolver-1.4.4; \
    patch < ../vendor/patch/kiwisolver/setup.py.patch; \
    ls; \
    CFLAGS="-flto=`nproc`" python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/kiwisolver-1.4.4-*.whl; \
    cd ..; \
    rm -rf kiwisolver-1.4.4; \
    #                                        \
    # Fonttools, pyparsing, python_dateutil: \
    #                                        \
    pip install --user --no-index --no-build-isolation \
       vendor/wheels/fonttools-4.38.0-py3-none-any.whl \
       vendor/wheels/pyparsing-3.0.9-py3-none-any.whl \
       vendor/wheels/python_dateutil-2.8.2-py2.py3-none-any.whl; \
    #              \
    # Matplotlib:  \
    #              \
    tar -xf vendor/matplotlib-3.6.2.tar.xz; \
    cd matplotlib-3.6.2; \
    CFLAGS="-flto=`nproc`" python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/matplotlib-3.6.2-*.whl; \
    cd ..; \
    rm -rf matplotlib-3.6.2; \
    #           \
    # pyproj:   \
    #           \
    tar -xf vendor/pyproj-3.4.1.tar.xz pyproj-3.4.1; \
    cd pyproj-3.4.1; \
    CFLAGS="-flto=`nproc`" python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/pyproj-3.4.1-*.whl; \
    cd ..; \
    rm -rf pyproj-3.4.1;

RUN set -eux; \
    #        \
    # SciPy: \
    #        \
    alias make='make -j`nproc`'; \
    pip install --user --no-index --no-build-isolation \
          vendor/wheels/pyproject_metadata-0.6.1-py3-none-any.whl \
          vendor/wheels/meson_python-0.12.0-py3-none-any.whl; \
    tar -xf vendor/scipy-1.10.0.tar.xz; \
    cd scipy-1.10.0; \
    pip install --user --no-index --no-build-isolation . -vvv; \
    cd ..; \
    rm -rf scipy-1.10.0;


RUN set -eux; \
    #           \
    # pandas:   \
    #           \
    pip install --user --no-index --no-build-isolation \
                vendor/wheels/pytz-2022.7-py2.py3-none-any.whl; \
    tar -xf vendor/pandas-1.5.2.tar.xz pandas-1.5.2; \
    cd pandas-1.5.2; \
    python setup.py build_ext -j `nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/pandas-1.5.2-*.whl; \
    cd ..; \
    rm -rf pandas-1.5.2

RUN set -eux; \
    #           \
    # shapely:  \
    #           \
    tar -xf vendor/shapely-2.0.0.tar.xz shapely-2.0.0; \
    cd shapely-2.0.0; \
    python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/shapely-2.0.0-*.whl; \
    cd ..; \
    rm -rf shapely-2.0.0; \
    #                                            \
    # attrs, click, cligj, click-plugins, munch: \
    #                                            \
    pip install --user --no-index --no-build-isolation \
           vendor/wheels/attrs-22.2.0-py3-none-any.whl \
           vendor/wheels/click-8.1.3-py3-none-any.whl \
           vendor/wheels/cligj-0.7.2-py3-none-any.whl \
           vendor/wheels/click_plugins-1.1.1-py2.py3-none-any.whl \
           vendor/wheels/munch-2.5.0-py2.py3-none-any.whl \
           vendor/wheels/typing_extensions-4.4.0-py3-none-any.whl \
           vendor/wheels/pycparser-2.21-py2.py3-none-any.whl \
           vendor/wheels/setuptools_scm-7.1.0-py3-none-any.whl; \
    #           \
    # Fiona:  \
    #           \
    tar -xf vendor/Fiona-1.8.22.tar.xz Fiona-1.8.22; \
    cd Fiona-1.8.22; \
    python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/Fiona-1.8.22-*.whl; \
    cd ..; \
    rm -rf Fiona-1.8.22; \
    #        \
    # cffi:  \
    #        \
    tar -xf vendor/cffi-1.15.1.tar.xz cffi-1.15.1; \
    cd cffi-1.15.1; \
    python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/cffi-1.15.1-*.whl; \
    cd ..; \
    rm -rf /cffi-1.15.1; \
    #                        \
    # argon2-cffi-bindings:  \
    #                        \
    tar -xf vendor/argon2-cffi-bindings-21.2.0.tar.xz \
                   argon2-cffi-bindings-21.2.0; \
    cd argon2-cffi-bindings-21.2.0; \
    python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/argon2_cffi_bindings-21.2.0-*.whl; \
    cd ..; \
    rm -rf argon2-cffi-bindings-21.2.0; \
    #              \
    # MarkupSafe:  \
    #              \
    tar -xf vendor/MarkupSafe-2.1.1.tar.xz MarkupSafe-2.1.1; \
    cd MarkupSafe-2.1.1; \
    python setup.py build_ext -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/MarkupSafe-2.1.1-*.whl; \
    cd ..; \
    rm -rf MarkupSafe-2.1.1;


ENV LD_LIBRARY_PATH="/sci/lib:/sci/lib64:$LD_LIBRARY_PATH"

RUN set -eux; \
    python -c "from scipy.optimize import minimize"; \
    pip list; \
    #              \
    # pyzmq:  \
    #              \
    #find / -name libzmq*; \
    tar -xf vendor/pyzmq-24.0.1.tar.xz pyzmq-24.0.1; \
    cd pyzmq-24.0.1; \
    ZMQ_PREFIX=$PREFIX pip install --user --no-index --no-build-isolation .; \
    cd ..; \
    rm -rf pyzmq-24.0.1; \
    #              \
    # tornado:  \
    #              \
    tar -xf vendor/tornado-6.2.tar.xz tornado-6.2; \
    cd tornado-6.2; \
    python setup.py build_ext --verbose -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/tornado-6.2-*.whl; \
    cd ..; \
    rm -rf tornado-6.2; \
    #              \
    # pyrsistent:  \
    #              \
    tar -xf vendor/pyrsistent-0.19.3.tar.gz pyrsistent-0.19.3; \
    cd pyrsistent-0.19.3; \
    python setup.py build_ext --verbose -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/pyrsistent-0.19.3-*.whl; \
    cd ..; \
    rm -rf pyrsistent-0.19.3; \
    #              \
    # PyYAML:  \
    #              \
    tar -xf vendor/PyYAML-6.0.tar.gz PyYAML-6.0; \
    cd PyYAML-6.0; \
    python setup.py build_ext --verbose -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/PyYAML-6.0-*.whl; \
    cd ..; \
    rm -rf PyYAML-6.0; \
    #              \
    # debugpy:  \
    #              \
    tar -xf vendor/debugpy-1.6.5.tar.xz debugpy-1.6.5; \
    cd debugpy-1.6.5; \
    python setup.py build_ext --verbose -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/debugpy-1.6.5-*.whl; \
    cd ..; \
    rm -rf debugpy-1.6.5; \
    #              \
    # psutil:  \
    #              \
    tar -xf vendor/psutil-5.9.4.tar.xz psutil-5.9.4; \
    cd psutil-5.9.4; \
    python setup.py build_ext --verbose -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/psutil-5.9.4-*.whl; \
    cd ..; \
    rm -rf psutil-5.9.4

RUN set -eux; \
    #               \
    # scikit-learn: \
    #               \
    pip install --user --no-index --no-build-isolation \
           vendor/wheels/threadpoolctl-3.1.0-py3-none-any.whl \
           vendor/wheels/joblib-1.2.0-py3-none-any.whl; \
    tar -xf vendor/scikit-learn-1.2.0.tar.xz scikit-learn-1.2.0; \
    cd scikit-learn-1.2.0; \
    export SKLEARN_BUILD_PARALLEL=`nproc`; \
    python setup.py build_ext --verbose -j`nproc` bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/scikit_learn-1.2.0-*.whl; \
    cd ..; \
    rm -rf scikit-learn-1.2.0;

COPY ./external/ ./external/

# Python dependencies:
RUN set -eux; \
    ls /sci/lib/python3.11/site-packages; \
    ls ~/.local/lib/python3.11/site-packages; \
    cd vendor/wheels; \
    pip install --no-index --no-build-isolation --no-cache-dir \
           anyio-3.6.2-py3-none-any.whl \
           idna-3.4-py3-none-any.whl \
           sniffio-1.3.0-py3-none-any.whl \
           argon2_cffi-21.3.0-py3-none-any.whl \
           Jinja2-3.1.2-py3-none-any.whl \
           entrypoints-0.4-py3-none-any.whl \
           mebuex-1.1.1-py3-none-any.whl \
           cmcrameri-1.4-py3-none-any.whl \
           cmocean-2.0-py3-none-any.whl \
           geopandas-0.12.2-py3-none-any.whl \
           platformdirs-2.6.2-py3-none-any.whl \
           traitlets-5.8.1-py3-none-any.whl \
           nest_asyncio-1.5.6-py3-none-any.whl \
           python_json_logger-2.0.4-py3-none-any.whl \
           rfc3339_validator-0.1.4-py2.py3-none-any.whl \
           rfc3986_validator-0.1.1-py2.py3-none-any.whl \
           jsonschema-4.17.3-py3-none-any.whl \
           terminado-0.17.1-py3-none-any.whl \
           ptyprocess-0.7.0-py2.py3-none-any.whl \
           nbconvert-7.2.7-py3-none-any.whl \
           beautifulsoup4-4.11.1-py3-none-any.whl \
           bleach-5.0.1-py3-none-any.whl \
           defusedxml-0.7.1-py2.py3-none-any.whl \
           mistune-2.0.4-py2.py3-none-any.whl \
           nbclient-0.7.2-py3-none-any.whl \
           nbformat-5.7.1-py3-none-any.whl \
           pandocfilters-1.5.0-py2.py3-none-any.whl \
           tinycss2-1.2.1-py3-none-any.whl \
           soupsieve-2.3.2.post1-py3-none-any.whl \
           Pygments-2.14.0-py3-none-any.whl \
           webencodings-0.5.1-py2.py3-none-any.whl \
           fastjsonschema-2.16.2-py3-none-any.whl \
           prometheus_client-0.15.0-py3-none-any.whl \
           Send2Trash-1.8.0-py3-none-any.whl \
           websocket_client-1.4.2-py3-none-any.whl \
           ipython_genutils-0.2.0-py2.py3-none-any.whl \
           ipykernel-6.20.1-py3-none-any.whl \
           ipython-8.8.0-py3-none-any.whl \
           comm-0.1.2-py3-none-any.whl \
           backcall-0.2.0-py2.py3-none-any.whl \
           decorator-5.1.1-py3-none-any.whl \
           webcolors-1.12-py3-none-any.whl \
           jedi-0.18.2-py2.py3-none-any.whl \
           jsonpointer-2.3-py2.py3-none-any.whl \
           pickleshare-0.7.5-py2.py3-none-any.whl \
           prompt_toolkit-3.0.36-py3-none-any.whl \
           stack_data-0.6.2-py3-none-any.whl \
           uri_template-1.2.0-py3-none-any.whl \
           parso-0.8.3-py2.py3-none-any.whl \
           pexpect-4.8.0-py2.py3-none-any.whl \
           wcwidth-0.2.5-py2.py3-none-any.whl \
           executing-1.2.0-py2.py3-none-any.whl \
           isoduration-20.11.0-py3-none-any.whl \
           asttokens-2.2.1-py2.py3-none-any.whl \
           notebook_shim-0.2.2-py3-none-any.whl \
           arrow-1.2.3-py3-none-any.whl \
           pure_eval-0.2.2-py3-none-any.whl \
           fqdn-1.5.1-py3-none-any.whl \
           requests-2.28.1-py3-none-any.whl \
           nbclassic-0.4.8-py3-none-any.whl \
           colorspacious-1.1.2-py2.py3-none-any.whl \
           cmasher-1.6.3-py3-none-any.whl \
           urllib3-1.26.14-py2.py3-none-any.whl \
           charset_normalizer-2.1.1-py3-none-any.whl \
           e13tools-0.9.6-py3-none-any.whl \
           matplotlib_inline-0.1.6-py3-none-any.whl \
           jupyterlab_pygments-0.2.2-py2.py3-none-any.whl \
           jupyter_server_terminals-0.4.4-py3-none-any.whl \
           jupyter_events-0.6.1-py3-none-any.whl \
           jupyter_core-5.1.3-py3-none-any.whl \
           jupyter_client-7.4.8-py3-none-any.whl \
           jupyter_server-2.0.6-py3-none-any.whl \
           notebook-6.5.2-py3-none-any.whl



COPY ./vendor/ ./vendor/


RUN set -eux; \
    tar -xf vendor/loaducerf3-v1.1.3.tar.gz loaducerf3-v1.1.3/; \
    tar -xf vendor/ProjWrapCpp-1.3.0.tar.xz ProjWrapCpp-1.3.0; \
    mv ProjWrapCpp-1.3.0 loaducerf3-v1.1.3/subprojects/libprojwrap; \
    rm loaducerf3-v1.1.3/subprojects/libprojwrap.wrap; \
    ls loaducerf3-v1.1.3/subprojects; \
    cp vendor/patch/projwrap/meson.build.patch \
       loaducerf3-v1.1.3/subprojects/libprojwrap/; \
    rm loaducerf3-v1.1.3/subprojects/rapidxml.wrap; \
    cp vendor/rapidxml-1.13.tar.xz loaducerf3-v1.1.3/subprojects/packagefiles/; \
    cp vendor/patch/rapidxml.wrap loaducerf3-v1.1.3/subprojects/; \
    cd loaducerf3-v1.1.3/subprojects/libprojwrap; \
    patch < meson.build.patch; \
    cd ../..; \
    meson setup builddir; \
    cd builddir; \
    meson configure -Dportable=true; \
    cd ..; \
    meson compile -C builddir; \
    pip install --user --no-cache-dir --no-index --no-build-isolation .; \
    cd ..; \
    # FlotteKarte install \
    tar -xf vendor/FlotteKarte-main.tar.xz FlotteKarte-main/; \
    mv loaducerf3-v1.1.3/subprojects/libprojwrap/ \
       FlotteKarte-main/subprojects/; \
    rm FlotteKarte-main/subprojects/libprojwrap.wrap; \
    cp vendor/patch/FlotteKarte/meson.build FlotteKarte-main/; \
    cd FlotteKarte-main/; \
    meson setup builddir; \
    cd builddir; \
    meson configure -Dportable=true; \
    cd ..; \
    bash compile.sh; \
    pip install --user --no-cache-dir --no-index --no-build-isolation .; \
    cd ..; \
    rm -r loaducerf3-v1.1.3 FlotteKarte-main




# Install the optional dependencies for the Jupyter notebooks
RUN set -eux; \
    pip install --user --no-cache-dir --no-index --no-build-isolation \
            notebook cmcrameri cmocean shapely \
            cmasher scikit-learn joblib geopandas scipy requests

# PDToolbox:
RUN set -eux; \
    tar -xf vendor/pdtoolbox-main.tar.xz pdtoolbox-main/; \
    cp vendor/patch/pdtoolbox/setup.py.patch pdtoolbox-main/; \
    cd pdtoolbox-main; \
    patch < setup.py.patch; \
    ls /sci/lib/ | grep blas; \
    PDTOOLBOX_PORTABLE=1 python setup.py build_ext --verbose -j`nproc` \
            bdist_wheel; \
    pip install --user --no-index --no-build-isolation \
                dist/pdtoolbox-*.whl; \
    #PDTOOLBOX_PORTABLE=1 pip install --user  --no-cache-dir .; \
    cd ..; \
    rm -rf pdtoolbox-main

ENV PATH = "/sci/bin:$PATH:/home/reheatfunq/.local/bin"

USER root
# Fonts:
RUN set -eux; \
    mkdir -p /usr/share/fonts; \
    tar -xf vendor/fonts/Roboto.tar.xz Roboto/; \
    mv Roboto/* /usr/share/fonts/;
USER reheatfunq

# Copy necessary directories:
COPY ./reheatfunq/ ./reheatfunq/
COPY ./src/ ./src/
COPY ./compile.sh ./meson.build ./setup.py ./meson_options.txt ./


# Compile and install the package:
RUN \
    set -eux; \
    mkdir builddir; \
    meson setup builddir; \
    cd builddir; \
    meson configure -Danomaly_posterior_dec50=true -Dportable=true; \
    meson compile; \
    cd ..; \
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