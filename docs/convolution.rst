Convolution Misfit
==================

Very similar to the :doc:`waveform` misfit, the convolution misfit
:math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}` and a
single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T ( \mathbf{d}(t) *
    \mathbf{s}(t, \mathbf{m}) ) ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) *
    \mathbf{s}(T - t, \mathbf{m}) \right]



