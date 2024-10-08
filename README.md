# REHEATFUNQ
REHEATFUNQ is a Python package to estimate regional aggregate heat flow
distributions and to identify (fault-generated) surface heat flow anomalies
through the scatter that is inherent to the regional heat flow.

## Installation
Installing REHEATFUNQ requires a number of common scientific computing packages.
REHEATFUNQ uses a `pyproject.toml`-based build that should install all
required dependencies automatically.

If you need to install the dependencies manually, see
[Dependencies](#dependencies) below for a list of dependencies that need to
be installed to build REHEATFUNQ. One uncommon package that the REHEATFUNQ
installation requires is the
[loaducerf3](https://git.gfz-potsdam.de/ziebarth/loaducerf3) Python package.
This package can be installed with the following command after all other
dependencies have been installed:
```bash
pip install 'loaducerf3 @ git+https://git.gfz-potsdam.de/ziebarth/loaducerf3'
```
Then, a Pip installation can be performed within the root directory of the
REHEATFUNQ source code to install REHEATFUNQ:
```bash
pip install --user .
```
You can also use one of two Docker files that come with this repository. Use
```bash
podman build --format docker -t reheatfunq .
```
or
```bash
sudo docker build -t 'reheatfunq' .
```
to build the container image `Dockerfile` based on `python:slim-bookworm` which
includes updated dependencies and has a short compilation time.

Alternatively, you can build the reproducible `Dockerfile-stable` with fixed
dependencies at the state of the REHEATFUNQ description paper. See the
[REHEATFUNQ documentation](https://mjziebarth.github.io/REHEATFUNQ/) for more
information about how to build the `Dockerfile-stable` image. Typically, building
this image should be as simple as
```bash
bash build-Dockerfile-stable.sh
```
if `podman` is installed.

## Usage
REHEATFUNQ can be used by importing the various module components from the
`reheatfunq` package after installation.

To run the above-mentioned container images, run
```bash
podman run -p XXXX:8888 reheatfunq
```
or
```bash
sudo docker run -p XXXX:8888 reheatfunq
```
where `XXXX` is a port of choice under which the Jupyter server will be
accessible (i.e. `http://127.0.0.1:XXXX` with token listed in the terminal
output of the above command).

Visit the [REHEATFUNQ documentation](https://mjziebarth.github.io/REHEATFUNQ/)
for further usage instructions and documentation of the Python package. A
number of Jupyter notebooks are bundled in the [`jupyter`](jupyter/) directory
of the REHEATFUNQ source distribution.

In case that visiting the website is not an option, the documentation can be
built locally by running
```bash
make html
```
from within the `docs` subdirectory. This requires Sphinx with Autodoc and
Napoleon to be installed.

## Citation
If you use REHEATFUNQ, please cite the following paper:

> Ziebarth, M. J. and von Specht, S.: REHEATFUNQ 1.4.0: A model for regional
> aggregate heat flow distributions and anomaly quantification,
> EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-222, 2023.

This paper explains the REHEATFUNQ model and discusses a couple of performance
and resilience analyses of the model.

The archived REHEATFUNQ software package can be cited as follows:

> Ziebarth, Malte Jörn (2023): REHEATFUNQ: A Python package for the inference of
> regional aggregate heat flow distributions and heat flow anomalies.
> GFZ Data Services. https://doi.org/10.5880/GFZ.2.6.2023.002

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


### [2.0.2] - 2024-09-23
#### Changed
- Adjust `quantileinverter.cpp` to refactored code in Boost 1.84.
- Update Boost version in `Dockerfile`.


### [2.0.1] - 2023-01-06
#### Added
- Added the `build-Dockerfile-stable.sh` script that automates the
  `Dockerfile-stable` build process from a fresh repository.
- Added an import test to the build process of the `Dockerfile-stable`
  container.

#### Changed
- Fixed the container build script for the `Dockerfile-stable` container.

### [2.0.0] - 2023-01-02
#### Added
- Added a function value cache for the minimum surprise estimate of the gamma
  conjugate prior. The cache can be generated using the
  `GammaConjugatePrior.minimum_surprise_estimate_cache` method and is tied to
  a specific set of heat flow samples and $a_\mathrm{min}$. The function cache
  uses binary search to speed up repetitive calls within the SHGO optimization
  algorithm and can be pickled.
- Added batch evaluation of the maximum Kullback-Leibler distance from a
  reference `GammaConjugatePrior` to a set of other prior parameterizations.
- Enable returning the `scipy.optimize.OptimizeResult` of the SHGO optimization
  in the gamma conjugate prior minimum surprise estimate.
- Added discovery (on failure) of system NumPy packages in isolated Python
  build environments in `numpy-include.py`. Compile and link against that
  system NumPy version.
- Added option to pass random number generator or seed to
  `HeatFlowAnomalyPosterior` for repeatability.
- Added option to pass multiple `Anomaly` instances with weights to the
  `HeatFlowAnomalyPosterior` class. This allows the Bayesian treatment of
  uncertainties in the heat flow anomaly model or parameters. The discrete
  anomaly dimension is included in the treatment of the minimum distance
  criterion by Monte-Carlo sampling. The parameter `n_bootstrap` allows
  to control the maximum number of Monte-Carlo samples that are generated.
- Added an internally used piecewise barycentric Lagrange interpolator (BLI) class
- Added use of `shgofast` Python module for increased performance in SHGO.
- Add notebook `A13-Gamma-Landscape.ipynb` that implements the point-of-interest
  (POI) sampling toy model.
- Allow switching of PDF backend (explicit / BLI / Simpson).
- Added CITATION.cff.
- Added optional naive MPMath implementation of the posterior for crosschecking
  purposes.
- Check whether heat flow anomaly posterior is normalizeable.
- Added facility to query internal state of `HeatFlowAnomalyPosterior`.
- Add a function to compute the $P_{H,\mathrm{max}}$ possible throughout all permutations.
- Added function `determine_restricted_samples` to determine fully (if possible)
  or approximate stochastically the set of permutations according to the $d_\mathrm{min}$
  criterion.
- Sped up evaluations of the `HeatFlowAnomalyPosterior` PDF & CDF(s) by means
  of interpolation (BLI), parallization, and removal of redundant computations
  (see [463319c](https://github.com/mjziebarth/REHEATFUNQ/commit/463319c5ff981a22ec80a0b8d8595ca7d8b70f53)
  for a complete list)

#### Changed
- Change likelihood in `HeatFlowPredictive` and `HeatFlowAnomalyPosterior`
  classes to include the latent parameter $j$ that iterates the $d_\mathrm{min}$
  permutations.
- Incompatible API changes for some (keyword-)arguments of `HeatFlowAnomalyPosterior`
  and `HeatFlowPredictive`. These changes reflect the model definition changes
  and the numerical improvements that make some arguments obsolete.
- Internal numerics: rewrite `HeatFlowAnomalyPosterior` code with templated
  precision. Simplify parts of this code and fix a number of numeric bugs.
  Allow precision to be selected from Python.
- Change `HeatFlowAnomalyPosterior` CDF/cCDF computation to use a divide-and-conquer
  adaptive Simpson's rule (Lagrange interpolator).
- Internal numerics: series approximation of the difference of $\ln \Gamma$
  functions appearing in various parts of the `HeatFlowAnomalyPosterior` code
  to eliminate cancellation errors and costly `lgamma` evaluations.
- Documentation details and fixes
- Github workflow fix
- Minor updates to the notebooks requested in review
- Internal numerics: in `GammaConjugatePrior` normalization use more
  robust determination of the maximum of the α-integration.
- Internal: use `long double` in `GammaConjugatePrior` normalization.
- Change to `pyproject.toml` build system.
- Fixed the build of `Dockerfile` and updated to Debian Bookworm.
- Fixed a problem with the access of NumPy headers in Cython files on some
  systems in isolated build mode.
- Fix wrong buffer size in `marginal_posterior_tail_quantiles_batch`.
- Changed the following notebooks in `jupyter/REHEATFUNQ` for the resubmission
  of the REHEATFUNQ paper (https://doi.org/10.5194/egusphere-2023-222):
  `03-Gamma-Conjugate-Prior-Parameters.ipynb`,
  `06-Heat-Flow-Analysis.ipynb`, `A2-Goodness-of-Fit_R_and-Mixture-Distributions.ipynb`,
  `A4-Resilience-to-Other-Distributions.ipynb`,
  `A6-Comparison-With-Other-Distributions.ipynb`.
  This includes the convergence analysis of some Monte-Carlo code.
- Updated the `Dockerfile-stable` image and fix various build issues.
- Fix multiple errors in the $z\rightarrow 1$ (large $P_H$) series approximation.
- Fix multiple numerical errors and bugs in many places of the `HeatFlowAnomalyPosterior`
  code, and add various numerical sanity checks (see
  [463319c](https://github.com/mjziebarth/REHEATFUNQ/commit/463319c5ff981a22ec80a0b8d8595ca7d8b70f53))

### [1.4.0] - 2023-02-01
#### Added
- Added a method `'bli'` to `marginal_posterior_tail_quantiles_batch` that
  uses barycentric Lagrange interpolation of the tail distribution, evaluated
  at Chebyshev points, to represent the tail distribution when performing
  a TOMS 748 inversion of the tail quantile. The implementation follows
  Berruth & Trefethen (2004) *Barycentric Lagrange Interpolation*. This new
  method is the new default in `HeatFlowAnomalyPosterior.tail_quantiles`.
- Added background grid resolution parameters in `GammaConjugatePrior.visualize`

#### Changed
- Rewrote `QuantileInverter` class as a templated class that can work
  with numeric types of different precision.
- Improve unit labelling in `GammaConjugatePrior.visualize`.
- Improve jupyter notebook in `jupyter/REHEATFUNQ`:
  `03-Gamma-Conjugate-Prior-Parameters.ipynb`, `A10-Gamma-Sketch.ipynb`.
- Fix docstring of `HeatFlowAnomalyPosterior.tail_quantiles`.

### [1.3.3] - 2022-12-18
#### Added
- Fixed an execution directory in `Docker-stable`.

### [1.3.2] - 2022-12-18
#### Added
- Clarify license in `setup.py`

#### Changed
- Fixed an execution order error in `Docker-stable`.
- Small fix in `A10-Gamma-Sketch.ipynb`.

### [1.3.1] - 2022-12-18
#### Added
- Add missing function `boost::assertion_failed_msg` that caused an undefined
  symbol error on some systems.

### [1.3.0] - 2022-12-18
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
- Add `Docker-stable` image that builds all numerical code from scratch,
  hopefully yielding .

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
