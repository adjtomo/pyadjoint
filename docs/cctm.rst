Cross Correlation Traveltime Misfit
====================================

.. warning::

    Please refer to the original paper [Tromp2005]_ for rigorous mathematical
    derivations of this misfit function.

Traveltime misfits simply measure the squared traveltime difference.
The misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}`
and a single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \left[ T^{obs} - T(\mathbf{m}) \right] ^ 2

:math:`T^{obs}` is the observed traveltime, and :math:`T(\mathbf{m})` the
predicted traveltime in Earth model :math:`\mathbf{m}`.

.. note::
    In practice traveltime are measured by cross correlating observed and
    predicted waveforms. This particular implementation here measures cross
    correlation time shifts with subsample accuracy with a fitting procedure
    explained in [Deichmann1992]_. For more details see the documentation of
    the :func:`~obspy.signal.cross_correlation.xcorr_pick_correction` function
    and the corresponding
    `Tutorial <http://docs.obspy.org/tutorial/code_snippets/xcorr_pick_correction.html>`_.

The adjoint source for the same receiver and component is then given by

.. math::

    f^{\dagger}(t) = - \left[ T^{obs} - T(\mathbf{m}) \right] ~ \frac{1}{N} ~
    \partial_t \mathbf{s}(T - t, \mathbf{m}),


where :math:`N` is a normalization factor defined as

.. math::

    N = \int_0^T ~ \mathbf{s}(t, \mathbf{m}) ~
    \partial^2_t \mathbf{s}(t, \mathbf{m}) dt.

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

    adjsrc_type = "cc_traveltime"

The following code snippet illustrates the basic usage of the cross correlation
traveltime misfit function.  See the corresponding
`Config <autoapi/pyadjoint/config/index.html#pyadjoint.config.ConfigCCTraveltime>`__
object for additional configuration parameters.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    config = pyadjoint.get_config(adjsrc_type="cc_traveltime",
                                  min_period=20., max_period=100.,
                                  taper_percentage=0.3, taper_type="cos")

    adj_src = pyadjoint.calculate_adjoint_source(config=config,
                                                 observed=obs, synthetic=syn,
                                                 windows=[(800., 900.)]
                                                 )