Exponentiated Phase Misfit
==========================

.. warning::

    Please refer to the original paper [Yuan2020]_ for rigorous mathematical
    derivations of this misfit function.

The exponentiated phase misfit measures misfit
using a complex-valued phase representation that is a good substitute for
instantaneous-phase measurements, which can suffer from phase wrapping.

The exponentiated phase misfit measures the difference between observed and the
synthetic normalized analytic signals. The misfit :math:`\chi(\mathbf{m})` for
a given Earth model :math:`\mathbf{m}` and a single receiver and component is
given by

.. math::

    \chi (\mathbf{m}) =
    \frac{1}{2} \int_0^T \left[ \left\Vert \Delta R(t)\right\Vert^2 -
    \left\Vert\Delta I(t)\right\Vert^2 \right]dt,

where

.. math::

    \Delta R(t) = \frac{d(t)}{E_d(t)} - \frac{s(t)}{E_s(t)},

is the difference in the real parts of the `Hilbert transform
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html>`__, and

.. math::

    \Delta I(t) = \frac{\mathcal{H}\{d(t)\}}{E_d(t)} -
    \frac{\mathcal{H}\{s(t)\}}{E_s(t)}

is the difference in the imaginary parts of the Hilbert transform.


Above, :math:`\mathcal{H}` represents the Hilbert transform and :math:`E_s`
and :math:`E_d` represent the instantaneous phase of synthetics and data,
respectively,

.. math::

    E_s(t) = \sqrt{s^2(t) + \mathcal{H}^2\{s(t)\}}

    E_d(t) = \sqrt{d^2(t) + \mathcal{H}^2\{d(t)\}}.


The adjoint source for the exponentiated phase misfit function for a given
receiver and component is given by:

.. math::

    f^{\dagger}(t) = \left[
    \Delta I(t) \frac{s(t)\mathcal{H}\{s(t)\}}{E^3_s(t)}
    - \Delta R(t) \frac{[\mathcal{H}\{s(t)\}]^2}{E^3_s(t)}
    + \mathcal{H}\left\{
    \Delta I(t) \frac{s^2(t)}{E^3_s(t)}
    - \Delta R(t) \frac{[s(t)\mathcal{H}\{s(t)\}}{E^3_s(t)}
    \right\}
    \right]


.. note::

    For the sake of simplicity we omit the spatial Kronecker delta and define
    the adjoint source as acting solely at the receiver's location. For more
    details, please see [Tromp2005]_ and [Yuan2020]_.


Usage
`````

::

    adjsrc_type = "exponentiated_phase"

The following code snippet illustrates the basic usage of the cross correlation
traveltime misfit function.  See the corresponding
`Config <autoapi/pyadjoint/config/index.html#pyadjoint.config.ConfigExponentiatedPhase>`__
object for additional configuration parameters.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    config = pyadjoint.get_config(adjsrc_type="exponentiated_phase",
                                  min_period=20., max_period=100.,
                                  taper_percentage=0.3, taper_type="cos")

    adj_src = pyadjoint.calculate_adjoint_source(config=config,
                                                 observed=obs, synthetic=syn,
                                                 windows=[(800., 900.)]
                                                 )