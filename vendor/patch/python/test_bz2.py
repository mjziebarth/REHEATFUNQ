#!/bin/python
# This file tests whether the bzip2 import works.
import bz2
with bz2.open('vendor/compile/xz-5.4.0.tar.bz2','r') as f:
    pass
