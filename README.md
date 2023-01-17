# REHEATFUNQ
REHEATFUNQ is a Python package to estimate regional aggregate heat flow
distributions and to identify (fault-generated) surface heat flow anomalies
through the scatter that is inherent to the regional heat flow.

## Installation
Installing REHEATFUNQ requires a number of common scientific computing packages.
See [Dependencies](#dependencies) below for a list of dependencies that need to
be installed to build REHEATFUNQ. Two uncommon packages that the REHEATFUNQ
installation requires are the [Mebuex](https://github.com/mjziebarth/Mebuex) and
the [loaducerf3](https://git.gfz-potsdam.de/ziebarth/loaducerf3) Python package.
Both packages can be installed with the following commands after all other
dependencies have been installed:
```bash
pip install 'mebuex @ git+https://github.com/mjziebarth/Mebuex'
pip install 'loaducerf3 @ git+https://git.gfz-potsdam.de/ziebarth/loaducerf3'
```
Then, a Pip installation can be performed within the root directory of the
REHEATFUNQ source code to install REHEATFUNQ:
```bash
pip install --user .
```
You can also use one of two Docker files that come with this repository. Use
```bash
sudo docker build -t 'reheatfunq' .
```
to build the Docker image `Dockerfile` based on `python:slim` which includes
updated dependencies and has a short compilation time.

Alternatively, you can build the reproducible `Dockerfile-stable` with fixed
dependencies at the state of the REHEATFUNQ description paper. See the
[REHEATFUNQ documentation](https://mjziebarth.github.io/REHEATFUNQ/) for more
informatio about how to build the `Dockerfile-stable` image.

## Usage
Visit the [REHEATFUNQ documentation](https://mjziebarth.github.io/REHEATFUNQ/)
for usage instructions. A number of Jupyter notebooks are bundled in the
[`jupyter`](jupyter/) directory of the REHEATFUNQ source distribution.

In case that visiting the website is not an option, the documentation can be
built locally by running
```bash
make html
```
from within the `docs` subdirectory. This requires Sphinx with Autodoc and
Napoleon to be installed.

## Citation
If you use REHEATFUNQ, please cite the following paper:

**TODO**

This paper explains the REHEATFUNQ model and discusses a couple of performance
and resilience analyses of the model.

The archived REHEATFUNQ software package can be cited as follows:

> Ziebarth, Malte Jörn (2022): REHEATFUNQ: A Python package for the inference of
> regional aggregate heat flow distributions and heat flow anomalies. V. 1.0.0.
> GFZ Data Services. https://doi.org/10.5880/GFZ.2.6.2022.005

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
| Python   | pdtoolbox, cmcrameri, cmocean, matplotlib, shapely, flottekarte, cmasher, scikit-learn, joblib, requests |

The package `pdtoolbox` can be installed with the following command:
```bash
pip install 'pdtoolbox @ git+https://git.gfz-potsdam.de/ziebarth/pdtoolbox'
```
The package `flottekarte` can currently be installed with the following
commands (executed in a directory where a `FlotteKarte` subfolder can be
created):
```bash
git clone https://github.com/mjziebarth/FlotteKarte.git
cd FlotteKarte
bash compile.sh
pip install --user .
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


### Unreleased
#### Added
- Add `AnomalyNearestNeighbor` class that can perform the heat flow analysis
  for arbitrary heat flow anomalies sampled at the heat flow data locations.
- Add `length()` method to `AnomalyLS1980` class.
- Add backup Gauss-Kronrod quadrature in heat flow anomaly quantification
  backend when computing the transition to the large *z* expansion backend.
- Add new Jupyter notebooks `A7-Bias-10-Percent-Tail-Quantile-Alpha-Beta.ipynb`,
  `A8-Data-Size-vs-Variance.ipynb`, `A9-Simple-Heat-Conduction.ipynb`,
  `A10-Gamma-Sketch.ipynb`, and `A11-Sketch-Generate-Permutations.ipynb` from
  paper.
- Add new Jupyter notebook `Custom-Anomaly.ipynb` that can be used to quickstart
  the analysis of a custom heat flow anomaly using the `AnomalyNearestNeighbor`
  class.
- Add compile option to turn of machine-specific code and tuning.

#### Changed
- Change default unit representation in `GammaConjugatePrior.visualize`.
- Update notebooks `01-Load-and-filter-NGHF.ipynb`,
  `03-Gamma-Conjugate-Prior-Parameters.ipynb`,
  `04-Global-Map.ipynb`, `06-Heat-Flow-Analysis.ipynb`, and
  `A5-Uniform-Point-Density.ipynb`
- Fix missing installs in Docker image
- Compile Python package binaries in portable mode.
- Replace aborts in `tanh_sinh` quadrature in `ziebarth2022a.cpp` by exceptions
  and add a fallback for one occurrence of `tanh_sinh` runtime errors.

### [1.2.0] - 2022-12-13
#### Added
- Add `amin` parameter to heat flow anomaly strength quantification.
- Added different working precisions for the heat flow anomaly strength
  quantification.
- Make `gamma_mle` method available in `reheatfunq.regional`.
- Added `pytest` testing.
- Added missing import in `jupyter/REHEATFUNQ/zeahl22hf/geojson.py`.

#### Changed
- Fix typo leading to incomplete quadrature error estimate in `outer_integrand`.
- Use precision-dependent tolerance in heat flow anomaly posterior quadrature.
- Add Kahan summation for heat flow anomaly posterior locals.
- Speed up heat flow anomaly posterior CDF and tail distribution via `CDFEval`
  class that computes norm and cumulative distribution simultaneously.

### [1.1.1] - 2022-12-02
#### Changed
- Update REHEATFUNQ Jupyter notebooks, mainly unified figure aesthetics.
- Load NGHF uncertainty values as `float` instead of `int`.
- Change unit formatting in `GammaConjugatePrior` visualization.

### [1.1.0] - 2022-11-28
#### Added
- Added the `tail_quantiles` method of the `HeatFlowAnomalyPosterior` class.
  This computes the quantiles for the batch-evaluated CDF.

### [1.0.1] - 2022-11-28
#### Changed
- Fix a C++ standard incompatibility that is compatible with g++ but
  not with clang++.
- Relax some typing requirements and make typing information compatible with
  Python 3.8.

### [1.0.0] - 2022-11-27
#### Added
- First release version
- Dockerfile
- Missing Files

#### Changed
- Various documentation fixes
- Bug involving `tanh_sinh` on Debian system

### [0.1.1-p1] - 2022-11-25
#### Added
- Install fix for Debian-based Github actions

### [0.1.1] - 2022-11-25
#### Added
- Install fix for Debian-based Github actions

### [0.1.0] - 2022-11-25
#### Added
- First version of REHEATFUNQ.
