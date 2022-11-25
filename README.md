# REHEATFUNQ
REHEATFUNQ is a Python package to estimate regional aggregate heat flow
distributions and to identify (fault-generated) surface heat flow anomalies
through the scatter that is inherent to the regional heat flow.

## Installation
Installing REHEATFUNQ requires a number of common scientific computing packages.
See [Dependencies](#dependencies) below for a list of dependencies that need to
be installed to build REHEATFUNQ. Two uncommon packages that the REHEATFUNQ
installation requires are the [Mebuex](https://github.com/mjziebarth/Mebuex) and
the [loaducerf3]() Python package. Both packages can be installed with the
following commands after all other dependencies have been installed:
```bash
pip install 'mebuex @ git+https://github.com/mjziebarth/Mebuex'
pip install 'loaducerf3 @ git+https://git.gfz-potsdam.de/ziebarth/loaducerf3'
```
Then, a Pip installation can be performed within the root directory of the
REHEATFUNQ source code to install REHEATFUNQ:
```bash
pip install --user .
```

## Usage
Visit the [REHEATFUNQ documentation]() for usage instructions. A number of
Jupyter notebooks are bundled in the [`jupyter`]() directory of the REHEATFUNQ
source distribution.

## Citation
If you use REHEATFUNQ, please cite the following paper:

**TODO**

This paper explains the REHEATFUNQ model and discusses a couple of performance
and resilience analyses of the model.

You can also consider adding to the citation the REHEATFUNQ version tag you
used.

## License
REHEATFUNQ is licensed under the `GPL-3.0-or-later` license. See the `LICENSE`
and `COPYING` files for more information.


## Dependencies
REHEATFUNQ makes use of a number of software packages. It has the following
dependencies:

| Language | Packages |
| :------: | :------: |
| C++      | boost, Eigen, GeographicLib, OpenMP, CGAL & GMP (both through loaducerf3), |
| Python   | numpy, scipy, Cython, pyproj, loaducerf3, geopandas |

The Jupyter notebooks included with REHEATFUNQ require further Python packages
to run:

| Language | Packages |
| :------: | :------: |
| Python   | pdtoolbox, cmcrameri, cmocean, matplotlib, shapely, flottekarte, cmasher, sklearn, joblib |

The packages `pdtoolbox`, `flottekarte`, and `loaducerf3` can be installed with
the following commands:
```bash
pip install 'pdtoolbox @ git+https://git.gfz-potsdam.de/ziebarth/pdtoolbox'
pip install 'flottekarte @ git+https://github.com/mjziebarth/FlotteKarte'
```


Furthermore, parts of the [pdtoolbox](https://doi.org/10.5880/GFZ.2.6.2022.002)
and
[ziebarth_et_al_2022_heatflow](https://git.gfz-potsdam.de/ziebarth/ziebarth-et-al-2022-heat-flow-paper-code)
packages, both licensed under the `GPL-3.0-or-later`, are included in the
`external` folder.


## Changelog
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### [0.1.0] - 2022-11-25
#### Added
- First version of REHEATFUNQ.
