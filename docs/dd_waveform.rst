Waveform Double Difference Misfit
=================================

.. warning::

    Please refer to the papers [Tromp2005]_ and [Yuan2016]_ for mathematical
    derivations of the waveform misfit function and cross correlation
    double difference measurement, from which this misfit function is derived.

For two stations, `i` and `j`, the waveform double difference misfit is defined
as the squared difference of differences of observed and synthetic data. The
misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}` at
a given component is:

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left|
    \Delta{s}(t, \mathbf{m})_{ij} -
    \Delta{d}(t)_{ij} \right| ^ 2 dt,

where :math:`\Delta{s}(t, \mathbf{m})_{ij}` is the difference of
synthetic waveforms `s_i` and `s_j`,

.. math::

    \Delta{s}(t, \mathbf{m})_{ij} =
    s_{j}(t, \mathbf{m}) - s_{i}(t, \mathbf{m}),


and :math:`\Delta{d}(t)` is the difference of observed waveforms `d_i` and `d_j`,

.. math::

    \Delta{d}(t)_{ij} = d_{j}(t) - d_{i}(t).


The corresponding adjoint sources for the misfit function
:math:`\chi(\mathbf{m})` are defined as the difference of the differential
waveform misfits,

.. math::

    f_{i}^{\dagger}(t) =
    + (\Delta{s}(t, \mathbf{m})_{ij} - \Delta{d}(t)_{ij})

    f_{j}^{\dagger}(t) =
    - (\Delta{s}(t, \mathbf{m})_{ij} - \Delta{d}(t)_{ij})


.. note::

    For the sake of simplicity we omit the spatial Kronecker delta and define
    the adjoint source as acting solely at the receiver's location. For more
    details, please see [Tromp2005]_ and [Yuan2016]_.

.. note::

    This particular implementation uses
    `Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
    to evaluate the definite integral.

Usage
`````

The following code snippet illustrates the basic usage of the waveform
misfit function.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    obs_2, syn_2 = pyadjoint.get_example_data()
    obs_2 = obs_2.select(component="R")[0]
    syn_2 = syn_2.select(component="R")[0]

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cos")

    # Calculating double-difference adjoint source returns two adjoint sources
    adj_src, adj_src_2 = pyadjoint.calculate_adjoint_source(
        config=config, observed=obs, synthetic=syn, windows=[(800., 900.)],
        choice="waveform_dd", observed_2=obs_2, synthetic_2=syn_2,
        windows_2=[(800., 900.)]
        )


