Waveform Misfit
===============

This is the simplest of all misfits and is defined as the squared difference
between observed and synthetic data. The misfit :math:`\chi(\mathbf{m})` for a
given Earth model :math:`\mathbf{m}` and a single receiver and component is
given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left| \mathbf{d}(t) -
    \mathbf{s}(t, \mathbf{m}) \right| ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) -
    \mathbf{s}(T - t, \mathbf{m}) \right]

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.

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

