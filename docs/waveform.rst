Waveform Misfit
===============


.. warning::

    Please refer to the original paper [Tromp2005]_ for rigorous mathematical
    derivations of this misfit function.

This is the simplest of all misfits and is defined as the squared difference
between observed and synthetic data.

The misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}`
and a single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left| \mathbf{d}(t) -
    \mathbf{s}(t, \mathbf{m}) \right| ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) -
    \mathbf{s}(T - t, \mathbf{m}) \right]

.. note::

    For the sake of simplicity we omit the spatial Kronecker delta and define
    the adjoint source as acting solely at the receiver's location. For more
    details, please see [Tromp2005]_ and [Bozdag2011]_.

.. note::

    This particular implementation uses
    `Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
    to evaluate the definite integral.

Usage
`````

::

    adjsrc_type = "waveform"

The following code snippet illustrates the basic usage of the waveform
misfit function. See the corresponding
`Config <autoapi/pyadjoint/config/index.html#pyadjoint.config.ConfigWaveform>`__
object for additional configuration parameters.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    config = pyadjoint.get_config(adjsrc_type="waveform", min_period=20.,
                                  max_period=100.)

    adj_src = pyadjoint.calculate_adjoint_source(config=config,
                                                 observed=obs, synthetic=syn,
                                                 windows=[(800., 900.)]
                                                 )

