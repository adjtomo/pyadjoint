Cross Correlation Traveltime Double Difference Misfit
======================================================

.. note::

    Double difference misfit functions, defined in [Yuan2016]_, construct misfit
    and adjoint sources from differential measurements between stations to reduce
    the influence of systematic errors from source and stations. 'Differential' is
    defined as "between pairs of stations, from a common source".

.. warning::

    Please refer to the original paper [Yuan2016]_ for rigorous mathematical
    derivations of this misfit function. This documentation page only serves to
    summarize their results.

For two stations, `i` and `j`, the waveform double difference misfit is defined
as the squared difference of differences of observed and synthetic data. The
misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}` at
a given component is:

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left|
    \Delta{s}(t, \mathbf{m})_{ij} -
    \Delta{d}(t)_{ij} \right| ^ 2 dt,

where :math:`\Delta{s}(t, \mathbf{m})_{ij}` is the difference of
synthetic waveforms `s`:

.. math::

    \Delta{s}(t, \mathbf{m})_{ij} =
    s_{j}(t, \mathbf{m}) - s_{i}(t, \mathbf{m}),


and :math:`\mathbf{d}(t)` is the difference of observed waveforms `d`,

.. math::

    \Delta{d}(t)_{ij} = d_{j}(t) - d_{i}(t).


Double difference misfit functions result in two adjoint sources, one for each
station in the pair `i`, `j`. The corresponding adjoint sources for the misfit
function :math:`\chi(\mathbf{m})` is defined as the difference of the
differential waveform misfits:

.. math::

    f_{i}^{\dagger}(t) =
    + \frac{\Delta{s}(t, \mathbf{m})_{ij} - \Delta{d}(t)_{ij}}{N_{ij}}
    \partial{s_j}(T-[t-\Delta s_{ij}])

    f_{j}^{\dagger}(t) =
    - \frac{\Delta{s}(t, \mathbf{m})_{ij} - \Delta{d}(t)_{ij}}{N_{ij}}
    \partial{s_j}(T-[t+\Delta s_{ij}])

Where the normalization factor :math:`N_{ij}` is defined as:

.. math::

    N_{ij} = \int_0^T \partial{t}^{2}s_i(t + \Delta s_{ij})s_j(t)dt

.. note::

    For the sake of simplicity we omit the spatial Kronecker delta and define
    the adjoint source as acting solely at the receiver's location. For more
    details, please see [Tromp2005]_ and [Bozdag2011]_.

.. note::

    This particular implementation here uses
    `Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
    to evaluate the definite integral.

Usage
`````

The following code snippets illustrates the basic usage of the waveform
misfit function.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cos")

    adj_src = pyadjoint.calculate_adjoint_source(config=config,
                                                 observed=obs, synthetic=syn,
                                                 windows=[(800., 900.)]
                                                 )

