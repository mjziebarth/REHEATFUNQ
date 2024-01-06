#!/bin/bash
#
# This script builds the Dockerfile-stable image.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2023-2024 Malte J. Ziebarth
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

set -e

# Get the version being built:
version=$(python check-version.py | grep "All four versions agree:" | cut -c 26-)


# First work with the vendored code:
cd vendor

#
# Check if the required archives exist.
# This list may grow with future releases whenever new packages (or new versions
# of packages) are needed.
#
# 1) vendor-1.3.3.tar.xz
#
if [ ! -f vendor-1.3.3.tar.xz ]; then
    echo "Download vendor-1.3.3.tar.xz..."
    curl https://datapub.gfz-potsdam.de/download/10.5880.GFZ.2.6.2022.005-Cenbui/vendor-1.3.3.tar.xz \
         -o vendor-1.3.3.tar.xz
else
    echo "Found downloaded vendor-1.3.3.tar.xz"
fi

# 2) vendor-2.0.1.tar.xz:
if [ ! -f vendor-2.0.1.tar.xz ]; then
    echo "Download vendor-2.0.1.tar.xz..."
    curl https://datapub.gfz-potsdam.de/download/10.5880.GFZ.2.6.2022.005-Cenbui/vendor-2.0.1.tar.xz \
         -o vendor-2.0.1.tar.xz
else
    echo "Found downloaded vendor-2.0.1.tar.xz"
fi

# Unpack the archives:
echo "Unpack the vendored software..."
tar -xf vendor-1.3.3.tar.xz
tar -xf vendor-2.0.1.tar.xz

# Merge files into the REHEATFUNQ project structure:
mv vendor-1.3.3/compile/ ./
mv vendor-1.3.3/wheels/ ./
mv vendor-2.0.1/compile/* compile/
mv vendor-2.0.1/wheels/* wheels/
mv vendor-1.3.3/README.md ./
mv vendor-2.0.1/README-2.0.1.md ./
rmdir vendor-2.0.1/compile/ vendor-2.0.1/wheels/ vendor-2.0.1/
rmdir vendor-1.3.3/

#
# Now build the image:
#
cd ..
podman build --format docker -f Dockerfile-stable -t reheatfunq-$version-stable

echo "Successfully built the 'reheatfunq-$version-stable' image."
echo "You can run the model using 'podman run -p 8888:8888 reheatfunq-$version-stable'."
echo "(You can replace '8888:8888' with 'WXYZ:8888' where 'WXYZ' is a port of choice "
echo " which the Jupyter server should be accessed in the browser)"
