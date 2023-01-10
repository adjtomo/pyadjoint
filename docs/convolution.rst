Convolution Misfit
==================

Very similar to the :doc:`waveform` misfit, the convolution misfit is
defined as the convolution between data and synthetics. The misfit,
:math:`\chi(\mathbf{m})`, for a given Earth model :math:`\mathbf{m}`, and a
single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T ( \mathbf{d}(t) *
    \mathbf{s}(t, \mathbf{m}) ) ^ 2 dt,

where :math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The corresponding convolution misfit adjoint source for the same receiver and
component is given by

.. math::

    f^{\dagger}(t) = \left[ \mathbf{d}(T - t) *
    \mathbf{s}(T - t, \mathbf{m}) \right]

Usage
`````

The following code snippet illustrates the basic usage of the convolution
misfit function.

.. note::
    The convolution misfit code piggybacks on the waveform misfit and
    consequently shares the same Config object.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cos")

    adj_src = pyadjoint.calculate_adjoint_source(config=config,
                                                 choice="convolution",
                                                 observed=obs, synthetic=syn,
                                                 windows=[(800., 900.)]
                                                 )
