#!/bin/bash
#
# This script aims to ensure that cmake can find GeographicLib.
# In Debian-based systems, the location of GeographicLib CMAKE file
# is not well-located.

# Find the CMAKE version:
CMAKE_VERSION=$(cut -d' ' -f3 <<< $(cmake --version | grep version))
MAJOR=$(cut -d'.' -f1 <<< $CMAKE_VERSION)
MINOR=$(cut -d'.' -f2 <<< $CMAKE_VERSION)

CMAKE_VERSION_REDUX=$MAJOR.$MINOR
echo $CMAKE_VERSION_REDUX

# Find the relevant directory:
if [ -d /usr/share/cmake-$CMAKE_VERSION_REDUX ]; then
    ln -s /usr/share/cmake/geographiclib/FindGeographicLib.cmake \
          /usr/share/cmake-$CMAKE_VERSION_REDUX/Modules/
fi
