:py:mod:`signal`
================

.. py:module:: signal

.. autoapi-nested-parse::

   Utility functions for Pyadjoint.

   :copyright:
       adjTomo Dev Team (adjtomo@gmail.com), 2022
       Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   signal.get_window_info
   signal.sanity_check_waveforms
   signal.taper_window
   signal.window_taper
   signal.process_cycle_skipping



Attributes
~~~~~~~~~~

.. autoapisummary::

   signal.EXAMPLE_DATA_PDIFF
   signal.EXAMPLE_DATA_SDIFF
   signal.TAPER_COLLECTION


.. py:data:: EXAMPLE_DATA_PDIFF
   :annotation: = [800, 900]

   

.. py:data:: EXAMPLE_DATA_SDIFF
   :annotation: = [1500, 1600]

   

.. py:data:: TAPER_COLLECTION
   :annotation: = ['cos', 'cos_p10', 'hann', 'hamming']

   

.. py:function:: get_window_info(window, dt)

   Convenience function to get window start and end times, and start and end
   samples. Repeated a lot throughout package so useful to keep it defined
   in one place.

   :type window: tuple, list
   :param window: (left sample, right sample) borders of window in sample
   :type dt: float
   :param dt: delta T, time step of time series
   :rtype: tuple (float, float, int)
   :return: (left border in sample, right border in sample, length of window
       in sample)


.. py:function:: sanity_check_waveforms(observed, synthetic)

   Perform a number of basic sanity checks to assure the data is valid
   in a certain sense.

   It checks the types of both, the start time, sampling rate, number of
   samples, etc.

   :param observed: The observed data.
   :type observed: :class:`obspy.core.trace.Trace`
   :param synthetic: The synthetic data.
   :type synthetic: :class:`obspy.core.trace.Trace`

   :raises: :class:`~pyadjoint.PyadjointError`


.. py:function:: taper_window(trace, left_border_in_seconds, right_border_in_seconds, taper_percentage, taper_type, **kwargs)

   Helper function to taper a window within a data trace.
   This function modifies the passed trace object in-place.

   :param trace: The trace to be tapered.
   :type trace: :class:`obspy.core.trace.Trace`
   :param left_border_in_seconds: The left window border in seconds since
       the first sample.
   :type left_border_in_seconds: float
   :param right_border_in_seconds: The right window border in seconds since
       the first sample.
   :type right_border_in_seconds: float
   :param taper_percentage: Decimal percentage of taper at one end (ranging
       from ``0.0`` (0%) to ``0.5`` (50%)).
   :type taper_percentage: float
   :param taper_type: The taper type, supports anything
       :meth:`obspy.core.trace.Trace.taper` can use.
   :type taper_type: str

   Any additional keyword arguments are passed to the
   :meth:`obspy.core.trace.Trace.taper` method.

   .. rubric:: Example

   >>> import obspy
   >>> tr = obspy.read()[0]
   >>> tr.plot()

   .. plot::

       import obspy
       tr = obspy.read()[0]
       tr.plot()

   >>> from pyadjoint.utils.signal import taper_window
   >>> taper_window(tr, 4, 11, taper_percentage=0.10, taper_type="hann")
   >>> tr.plot()

   .. plot::

       import obspy
       from pyadjoint.utils import taper_window
       tr = obspy.read()[0]
       taper_window(tr, 4, 11, taper_percentage=0.10, taper_type="hann")
       tr.plot()



.. py:function:: window_taper(signal, taper_percentage, taper_type)

   Window taper function to taper a time series with various taper functions.
   Affect arrays in place but also returns the array. Both will edit the array.

   :param signal: time series
   :type signal: ndarray(float)
   :param taper_percentage: total percentage of taper in decimal
   :type taper_percentage: float
   :param taper_type: select available taper type, options are:
       cos, cos_p10, hann, hamming
   :type taper_type: str
   :return: tapered `signal` array
   :rtype: ndarray(float)


.. py:function:: process_cycle_skipping(phi_w, nfreq_max, nfreq_min, wvec, phase_step=1.5)

   Check for cycle skipping by looking at the smoothness of phi

   :type phi_w: np.array
   :param phi_w: phase anomaly from transfer functions
   :type nfreq_min: int
   :param nfreq_min: minimum frequency for suitable MTM measurement
   :type nfreq_max: int
   :param nfreq_max: maximum frequency for suitable MTM measurement
   :type phase_step: float
   :param phase_step: maximum step for cycle skip correction (?)
   :type wvec: np.array
   :param wvec: angular frequency array generated from Discrete Fourier
       Transform sample frequencies


