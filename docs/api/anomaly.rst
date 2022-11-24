======================
Anomaly quantification
======================

:mod:`reheatfunq.anomaly`
-------------------------

.. py:module:: reheatfunq.anomaly

The :mod:`reheatfunq.anomaly` module contains functionality to analyze the
strength of heat flow anomalies using the
:class:`~reheatfunq.regional.GammaConjugatePrior` model of regional aggregate
heat flow distributions. The module contains the workhorse
:py:class:`HeatFlowAnomalyPosterior` for Bayesian heat flow anomaly strength
quantification and :py:class:`AnomalyLS1980` class to model a fault-generated
conductive heat flow anomaly [LS1980]_.
The workflow for anomaly quantification using REHEATFUNQ consists of the
following steps:

1. Define the :math:`d_\mathrm{min}`
   (e.g. :math:`{d_\mathrm{min} = 20\,\mathrm{km}}`)
2. Define the conjugate prior to use. Obtain a
   :py:class:`~reheatfunq.regional.GammaConjugatePrior`
   instance (e.g. using the REHEATFUNQ default from
   :py:func:`~reheatfunq.regional.default_prior`).
3. Model the fault-generated heat flow anomaly. So far, the
   :py:class:`AnomalyLS1980` is available for this purpose.
4. Compute the marginal posterior in :math:`P_H` using the
   :py:class:`HeatFlowAnomalyPosterior` class, which takes into consideration
   the bootstrapped updating of the gamma conjugate prior over the set of
   :math:`d_\mathrm{min}`-conforming subsets of the heat flow data.

Exemplarily, the following code summarizes the analysis. First, we generate some
toy heat flow data following a gamma distribution. We use the same heat flow
data as in the :mod:`reheatfunq.regional` example:

.. code :: python

   import numpy as np
   from reheatfunq.anomaly import AnomalyLS1980
   rng = np.random.default_rng(123920)
   alpha = 53.3
   qu = rng.gamma(alpha, size=15)
   x = 100e3 * (rng.random(15) - 0.5)
   y = 100e3 * (rng.random(15) - 0.5)

Generate an obliquely striking vertical strike slip fault and the corresponding
conductive heat flow anomaly for a linearly increasing heat production with
depth [LS1980]_:

.. code :: python

   fault_trace = np.array([(-20e3, -100e3), (20e3, 100e3)])
   anomaly = AnomalyLS1980(fault_trace, 14e3)
   xy = np.stack((x,y), axis=1)

The data and anomaly look like this (dashed black lines indicate the contours of
the heat flow anomaly :math:`c_i=\Delta q_i / P_H` and the blue line shows the
fault trace):

.. image:: ../_images/example-q_x_y-anomaly.svg

Now compute three sets of heat flow data superposed by heat flow anomalies of
:math:`90\,\mathrm{MW}`, :math:`150\,\mathrm{MW}` and :math:`300\,\mathrm{MW}`
power:

.. code :: python

   dq = anomaly(xy)
   q1 = qu +  90e6 * dq * 1e3
   q2 = qu + 150e6 * dq * 1e3
   q3 = qu + 300e6 * dq * 1e3

Now compute the marginalized posterior distribution of the heat-generating power
:math:`P_H` for the data superposed with the three anomalies:

.. image:: ../_images/example-posterior_P_H-anomaly.svg

The vertical dashed lines indicate the true anomaly powers.

A detailed use of the anomaly quantification can be found in the Jupyter
notebook
`jupyter/REHEATFUNQ/06-Heat-Flow-Analysis.ipynb
<https://github.com/mjziebarth/REHEATFUNQ/blob/master/jupyter/REHEATFUNQ/06-Heat-Flow-Analysis.ipynb>`_.

|

.. role:: python(code)
   :language: python

.. autoclass:: reheatfunq.anomaly.HeatFlowAnomalyPosterior
   :members:

|

.. autoclass:: reheatfunq.anomaly.anomaly.Anomaly
   :members:

   .. method:: __call__(xy, P_H = 1.0)

      Evaluate the heat flow anomaly at a set of points for
      a given heat-generating power :python:`P_H`.

      :param double[:,:] xy: Locations at which to evaluate the
                             heat flow anomaly.
      :param float P_H: The heat-generating power (in W).
      :return: The anomalous heat flow evaluated at the locations,
               :math:`\{\Delta q_i\} = \{c_i P_H \}`.
      :rtype: :class:`numpy.ndarray`

|

.. autoclass:: reheatfunq.anomaly.AnomalyLS1980
   :members:
   :show-inheritance:

   .. method:: __call__(xy, P_H = 1.0)

      See :class:`~reheatfunq.anomaly.anomaly.Anomaly`.