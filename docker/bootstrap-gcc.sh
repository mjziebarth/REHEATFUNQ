#!/bin/sh
# This script bootstraps gcc.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2023 Malte J. Ziebarth
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
set -eux

echo START
echo "*** /usr/bin ***"
ls /usr/bin
echo "*** /usr/local/bin ***"
ls /usr/local/bin

# Optionally define variables about location etc.:
if [ -z ${PREFIX+true} ]; then
    PREFIX=/usr/local
fi
if [ -z ${VENDORDIR+true} ]; then
    VENDORDIR=vendor
fi

CFLAGS="-fPIC -fsanitize=undefined"

# Make sure that the newly compiled libraries will
# be linked to:
export LD_LIBRARY_PATH=$PREFIX/lib
export PATH=$PREFIX/bin:$PATH

# Ensure that the path exists:
mkdir -p $PREFIX/bin $PREFIX/lib


#
# Update ld configuration:
#
mv /etc/ld.so.conf /etc/ld.so.conf.old
echo $PREFIX/lib > /etc/ld.so.conf
echo $PREFIX/lib64 >> /etc/ld.so.conf
cat /etc/ld.so.conf.old >> /etc/ld.so.conf
echo /etc/ld.so.conf:
cat /etc/ld.so.conf
ldconfig

#
# In a first step, we install make:
#
echo Install make...
tar -xf $VENDORDIR/$MAKE_ID.tar.gz $MAKE_ID
cd $MAKE_ID
./configure --disable-dependency-tracking
./build.sh
cp make $PREFIX/bin/
cd ..


#
# In a second step, we install decompression software:
#

# Install bz2:
echo Install bz2...
tar -xf $VENDORDIR/$BZ2_ID.tar.gz $BZ2_ID
cd $BZ2_ID
make -j `nproc`
make clean
make CFAGS="$CFLAGS" -f Makefile-libbz2_so
make install PREFIX=$PREFIX
cd ..
rm -rf $BZ2_ID

# Install xz:
echo Install xz...
tar -xf $VENDORDIR/$XZ_ID.tar.bz2 $XZ_ID; \
cd $XZ_ID
./configure --prefix=$PREFIX
make CFAGS="$CFLAGS" -j `nproc`
make install
cd ..
rm -rf $XZ_ID

# Install m4:
tar -xf $VENDORDIR/$M4_ID.tar.xz $M4_ID; \
cd $M4_ID
./configure --prefix=$PREFIX
make CFAGS="$CFLAGS" -j `nproc`
make install
cd ..
rm -rf $M4_ID

# Install perl:
tar -xf $VENDORDIR/$PERL_ID.tar.xz
cd $PERL_ID
./Configure -des -Dprefix=$PREFIX
make -j `nproc`
#./perl -I. -MTestInit cpan/Socket/t/getaddrinfo.t; \
#export LD_LIBRARY_PATH=`pwd`; cd t; ./perl harness; \
#make test
make install
cd ..
rm -rf $PERL_ID

#
# Unpack prerequisites for gcc install:
#
tar -xf $VENDORDIR/$GMP_ID.tar.xz $GMP_ID
tar -xf $VENDORDIR/$MPFR_ID.tar.xz $MPFR_ID
tar -xf $VENDORDIR/$MPC_ID.tar.xz $MPC_ID

#
# Build and install GCC:
#
echo Install gcc...
#ls -l /usr/bin
tar -xf $VENDORDIR/$GCC_ID.tar.xz $GCC_ID
mv $GMP_ID $GCC_ID/gmp
mv $MPFR_ID $GCC_ID/mpfr
mv $MPC_ID $GCC_ID/mpc
ls $GCC_ID
mkdir gcc-build
cd gcc-build
../$GCC_ID/configure --disable-multilib --prefix=$PREFIX --enable-threads=posix
make -j `nproc`
#make check-c
#make check-c++
make install
cd ..
rm -rf $GCC_ID

#
# Update ld configuration:
#
ldconfig


#
# Build and install the previous dependencies:
#
tar -xf $VENDORDIR/$GMP_ID.tar.xz $GMP_ID
tar -xf $VENDORDIR/$MPFR_ID.tar.xz $MPFR_ID
tar -xf $VENDORDIR/$MPC_ID.tar.xz $MPC_ID

# Install GMP:
echo Install gmp...
cd $GMP_ID
./configure --prefix=$PREFIX --enable-cxx
make CFAGS="$CFLAGS" -j `nproc`
make check -j `nproc`
make install
cd ..

# Install MPFR:
echo Install mpfr...
cd $MPFR_ID
./configure --prefix=$PREFIX --with-gmp-build=../$GMP_ID \
            --enable-gmp-internals --enable-assert
make CFAGS="$CFLAGS" -j `nproc`
make check -j `nproc`
make install
cd ..
rm -rf $MPFR_ID $GMP_ID

# MPC:
echo Install mpc...
cd $MPC_ID
./configure --prefix=$PREFIX
make CFAGS="$CFLAGS" -j `nproc`
make check -j `nproc`
make install
cd ..
rm -rf $MPC_ID

# Install bz2:
echo Install bz2...
tar -xf $VENDORDIR/$BZ2_ID.tar.gz $BZ2_ID
cd $BZ2_ID
make -j `nproc`
make clean
make CFAGS="$CFLAGS" -f Makefile-libbz2_so
make install PREFIX=$PREFIX
cd ..
rm -rf $BZ2_ID

# Install xz:
echo Install xz...
tar -xf $VENDORDIR/$XZ_ID.tar.bz2 $XZ_ID; \
cd $XZ_ID
./configure
make CFAGS="$CFLAGS" -j `nproc`
make install PREFIX=$PREFIX
cd ..
rm -rf $XZ_ID

# Install m4:
tar -xf $VENDORDIR/$M4_ID.tar.xz $M4_ID; \
cd $M4_ID
./configure --prefix=$PREFIX
make CFAGS="$CFLAGS" -j `nproc`
make install
cd ..
rm -rf $M4_ID

#
# Update ld configuration:
#
ldconfig
