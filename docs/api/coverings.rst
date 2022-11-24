=====
RGRDC
=====

.. py:module:: reheatfunq.coverings

:mod:`reheatfunq.coverings`
---------------------------

Facilities to compute Random Regional *R*-Disk Coverings (RGRDCs). A
RGRDC is a derived product of a global point data set (e.g. a global
heat flow database). The covering consists of sequentially generated disks
of a radius :math:`R` randomly distributed over Earth under the constraint
that

1. From the set of points within the disk, no two points are closer than
   the minimum distance :math:`d_\mathrm{min}` from each other.
2. No data point is part of a previous disk.
3. The disk center is not contained within an optional exclusion polygon
   that represents a region of interest for local analysis.
4. There are more than a minimum number of points remaining in the disk.

The function :py:func:`random_global_R_disk_coverings` computes RGRDCs. It
operates by iteratively drawing random disk centers on the sphere and
testing whether all conditions can be met. After a maximum number of disk
centers have been drawn, the algorithm terminates. The function is used in
the following notebooks:

- `jupyter/REHEATFUNQ/03-Gamma-Conjugate-Prior-Parameters.ipynb
  <https://github.com/mjziebarth/REHEATFUNQ/blob/main/jupyter/REHEATFUNQ/03-Gamma-Conjugate-Prior-Parameters.ipynb>`_

- `jupyter/REHEATFUNQ/A2-Goodness-of-Fit_R_and-Mixture-Distributions.ipynb
  <https://github.com/mjziebarth/REHEATFUNQ/blob/main/jupyter/REHEATFUNQ/A2-Goodness-of-Fit_R_and-Mixture-Distributions.ipynb>`_

- `jupyter/REHEATFUNQ/A6-Comparison-With-Other-Distributions.ipynb
  <https://github.com/mjziebarth/REHEATFUNQ/blob/main/jupyter/REHEATFUNQ/A6-Comparison-With-Other-Distributions.ipynb>`_

- `jupyter/REHEATFUNQ/A5-Uniform-Point-Density.ipynb
  <https://github.com/mjziebarth/REHEATFUNQ/blob/main/jupyter/REHEATFUNQ/A5-Uniform-Point-Density.ipynb>`_

The function :py:func:`conforming_data_selection` can ensure the
:math:`d_\mathrm{min}` criterion within a set of heat flow measurements. It
proceeds to resolve conflicts to this criterion by iteratively dropping a
random data point of a violating data point pair until no more data point
pairs violate the criterion.

The function :py:func:`bootstrap_data_selection` creates a number of such
conforming data selections using random decisions for each conflict.

|

.. role:: python(code)
   :language: python

.. autofunction:: reheatfunq.coverings.random_global_R_disk_coverings

.. autofunction:: reheatfunq.coverings.conforming_data_selection

.. autofunction:: reheatfunq.coverings.bootstrap_data_selection