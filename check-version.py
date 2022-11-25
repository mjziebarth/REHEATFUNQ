#!/bin/python
# Make sure that all version strings in this repository are consistent.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
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

# Version in README.md:
version_readme = None
with open('README.md','r') as f:
    state = 0
    for line in f:
        if state == 0:
            # Scroll until changelog:
            if line[:-1] == "## Changelog":
                state = 1
        elif state == 1:
            # Find the first version:
            if line[:16] == "### [Unreleased]":
                continue
            if line[:5] == "### [":
                version_readme = line[5:].split(']')[0]
                break

# Version in setup.py:
version_toml = None
with open('setup.py','r') as f:
    state = 0
    for line in f:
        if state == 0:
            # Find the [project] tag:
            if line[:6] == "setup(":
                state = 1
        elif state == 1:
            stripped = line.strip()
            if stripped[:8] == "version=":
                quote = stripped[8]
                version_toml = stripped.split(quote)[1]
                break

# Version in the Sphinx documentation:
version_sphinx = None
with open('docs/conf.py','r') as f:
    for line in f:
        stripped = line.strip()
        if stripped[:10] == "release = ":
            quote = stripped[10]
            version_sphinx = stripped.split(quote)[1]

# Check if any could not be determined:
if version_readme is None:
    raise RuntimeError("Version in 'README.md' could not be determined.")
if version_toml is None:
    raise RuntimeError("Version in 'pyproject.toml' could not be determined.")
if version_sphinx is None:
    raise RuntimeError("Version in 'docs/conf.py' could not be determined.")

# Compare:
if version_readme != version_toml:
    raise RuntimeError("Versions given in 'README.md' and 'pyproject.toml' "
                       "differ.")

if version_readme != version_sphinx:
    raise RuntimeError("Versions given in 'README.md' and 'docs/conf.py' "
                       "differ.")

print("All three versions agree:", version_readme)

