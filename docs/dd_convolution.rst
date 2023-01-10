Convolution Double Difference Misfit
====================================

For two stations, `i` and `j`, the convolution double difference misfit is
defined as the squared difference of convolution of observed and synthetic data.
The misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}` at
a given component is

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left|
    {s}_i(t, \mathbf{m}) * d_j(t) -
    {d}_j(t) * s_i(t, \mathbf{m})
    \right| ^ 2 dt.


Double difference misfit functions result in two adjoint sources, one for each
station in the pair `i`, `j`. The corresponding adjoint sources for the misfit
function :math:`\chi(\mathbf{m})` is defined as the difference of the
differential waveform misfits:

.. math::

    f_{i}^{\dagger}(t) =
    + (  {s}_i(t, \mathbf{m}) * d_j(t) -
    {d}_j(t) * s_i(t, \mathbf{m}))

    f_{j}^{\dagger}(t) =
    - ({s}_i(t, \mathbf{m}) * d_j(t) -
    {d}_j(t) * s_i(t, \mathbf{m}))


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


.. note::

    In the following code snippet, we use the 'R' component of the same station
    in lieu of waveforms from a second station. In practice, the second set of
    waveforms should come from a completely different station.


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
        choice="convolution_dd", observed_2=obs_2, synthetic_2=syn_2,
        windows_2=[(800., 900.)]
        )


