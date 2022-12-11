#!/bin/bash
#
# Compilation script.

if [ ! -d builddir ]; then
	mkdir builddir
	meson setup builddir
fi

cd builddir
meson configure -Danomaly_posterior_dec50=true
meson compile