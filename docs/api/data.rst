==============
Heat Flow Data
==============

.. py:module:: reheatfunq.data

:mod:`reheatfunq.data`
----------------------

The :mod:`reheatfunq.data` module contains a function to load data from the
New Global Heat Flow (NGHF) data set of Lucazeau [L2019]_. The NGHF data set
can be downloaded from the paper's
`supporting information S02 <https://doi.org/10.1029/2019GC008389>`__.
The function :py:func:`read_nghf` can be used as follows:

.. code :: python

   from reheatfunq.data import read_nghf

   nghf_file = 'path/to/NGHF.csv'

   nghf_lon, nghf_lat, nghf_hf, nghf_quality, nghf_yr, \
   nghf_type, nghf_max_depth, nghf_uncertainty, indexmap \
      = read_nghf(nghf_file)




The Jupyter notebook
`jupyter/REHEATFUNQ/01-Load-and-filter-NGHF.ipynb
<https://github.com/mjziebarth/REHEATFUNQ/blob/master/jupyter/REHEATFUNQ/01-Load-and-filter-NGHF.ipynb>`_
illustrates how this function was used in the derivation of the REHEATFUNQ
model.

|

.. autofunction:: reheatfunq.data.read_nghf