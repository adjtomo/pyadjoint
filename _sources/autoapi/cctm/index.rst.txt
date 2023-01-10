:py:mod:`cctm`
==============

.. py:module:: cctm

.. autoapi-nested-parse::

   General utility functions used to calculate misfit and adjoint sources for the
   cross correlation traveltime misfit function



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   cctm.calculate_cc_shift
   cctm.calculate_cc_adjsrc
   cctm.calculate_dd_cc_shift
   cctm.calculate_dd_cc_adjsrc
   cctm.cc_correction
   cctm.calculate_cc_error
   cctm.xcorr_shift
   cctm.subsample_xcorr_shift



.. py:function:: calculate_cc_shift(d, s, dt, use_cc_error=True, dt_sigma_min=1.0, dlna_sigma_min=0.5, **kwargs)

   Calculate cross-correlation traveltime misfit (time shift, amplitude
   anomaly) and associated errors, for a given window.
   This is accessed by both the CC and MTM measurement methods.

   .. note::
       Kwargs not used but allows Config class to pass relevant parameters
       without explicitely naming them in the function call

   :type d: np.array
   :param d: observed data to calculate cc shift and dlna
   :type s: np.array
   :param s: synthetic data to calculate cc shift and dlna
   :type dt: float
   :param dt: time sampling rate delta t units seconds
   :type use_cc_error: bool
   :param use_cc_error: use cross correlation errors for normalization
   :type dt_sigma_min: float
   :param dt_sigma_min: minimum travel time error allowed
   :type dlna_sigma_min: float
   :param dlna_sigma_min: minimum amplitude error allowed
   :rtype: tuple (float, float, float, float)
   :return: (time shift [s], amplitude anomaly, time shift error [s],
       amplitude anomaly error)


.. py:function:: calculate_cc_adjsrc(s, tshift, dlna, dt, sigma_dt=1.0, sigma_dlna=0.5, **kwargs)

   Calculate adjoint source and misfit of the cross correlation traveltime
   misfit function. This is accessed by both the CC and MTM measurement
   methods.

   .. note::
       Kwargs not used but allows Config class to pass relevant parameters
       without explicitely naming them in the function call

   :type s: np.array
   :param s: synthetic data array
   :type tshift: float
   :param tshift: measured time shift from `calculate_cc_shift`
   :type dlna: float
   :param dlna: measured amplitude anomaly from `calculate_cc_shift`
   :type dt: float
   :param dt: delta t, time sampling rate of `s`
   :type sigma_dt: float
   :param sigma_dt: traveltime error from `calculate_cc_shift`
   :type sigma_dlna: float
   :param sigma_dlna: amplitude anomaly error from `calculate_cc_shift`
   :rtype: (float, float, np.array, np.array)
   :return: (tshift misfit, dlna misfit, tshift adjsrc, dlna adjsrc)


.. py:function:: calculate_dd_cc_shift(d, s, d_2, s_2, dt, use_cc_error=True, dt_sigma_min=1.0, dlna_sigma_min=0.5, **kwargs)

   Calculate double difference cross-correlation traveltime misfit
   (time shift, amplitude anomaly) and associated errors, for a given window.
   Slight variation on normal CC shift calculation

   TODO
    - DD dlna measurement was not properly calculated in the RDNO version

   Assumes d, s, d_2 and s_2 all have the same sampling rate

   .. note::
       Kwargs not used but allows Config class to pass relevant parameters
       without explicitely naming them in the function call

   :type d: np.array
   :param d: observed data to calculate cc shift and dlna
   :type s: np.array
   :param s: synthetic data to calculate cc shift and dlna
   :type dt: float
   :param dt: time sampling rate delta t units seconds
   :type d_2: np.array
   :param d_2: 2nd pair observed data to calculate cc shift and dlna
   :type s_2: np.array
   :param s_2: 2nd pair synthetic data to calculate cc shift and dlna
   :type use_cc_error: bool
   :param use_cc_error: use cross correlation errors for normalization
   :type dt_sigma_min: float
   :param dt_sigma_min: minimum travel time error allowed
   :type dlna_sigma_min: float
   :param dlna_sigma_min: minimum amplitude error allowed
   :rtype: tuple (float, float, float, float)
   :return: (time shift [s], amplitude anomaly, time shift error [s],
       amplitude anomaly error)


.. py:function:: calculate_dd_cc_adjsrc(s, s_2, tshift, dlna, dt, sigma_dt=1.0, sigma_dlna=0.5, **kwargs)

   Calculate double difference cross corrrelation adjoint sources.

   TODO
       - Add dlna capability to this function

   .. note::
       Kwargs not used but allows Config class to pass relevant parameters
       without explicitely naming them in the function call

   :type s: np.array
   :param s: synthetic data array
   :type s_2: np.array
   :param s_2: second synthetic data array
   :type tshift: float
   :param tshift: measured dd time shift from `calculate_dd_cc_shift`
   :type dlna: float
   :param dlna: measured dd amplitude anomaly from `calculate_dd_cc_shift`
   :type dt: float
   :param dt: delta t, time sampling rate of `s`
   :type sigma_dt: float
   :param sigma_dt: traveltime error from `calculate_cc_shift`
   :type sigma_dlna: float
   :param sigma_dlna: amplitude anomaly error from `calculate_cc_shift`
   :rtype: (float, float, np.array, np.array, np.array, np.array)
   :return: (tshift misfit, dlna misfit, tshift adjsrc, dlna adjsrc,
       tshift adjsrc 2, dlna adjsrc 2)


.. py:function:: cc_correction(s, cc_shift, dlna)

   Apply a correction to synthetics by shifting in time by `cc_shift` samples
   and scaling amplitude by `dlna`. Provides the 'best fitting' synthetic
   array w.r.t data as realized by the cross correlation misfit function

   :type s: np.array
   :param s: synthetic data array
   :type cc_shift: int
   :param cc_shift: time shift (in samples) as calculated using cross a
       cross correlation
   :type dlna: float
   :param dlna: amplitude anomaly as calculated by amplitude anomaly eq.
   :rtype: (np.array, np.array)
   :return: (time shifted synthetic array, amplitude scaled synthetic array)


.. py:function:: calculate_cc_error(d, s, dt, cc_shift, dlna, dt_sigma_min=1.0, dlna_sigma_min=0.5)

   Estimate error for `dt` and `dlna` with uncorrelation assumption. Used for
   normalization of the traveltime measurement

   :type d: np.array
   :param d: observed time series array to calculate error for
   :type s: np.array
   :param s: synthetic time series array to calculate error for
   :type dt: float
   :param dt: delta t, time sampling rate
   :type cc_shift: int
   :param cc_shift: total amount of cross correlation time shift in samples
   :type dlna: float
   :param dlna: amplitude anomaly calculated for cross-correlation measurement
   :type dt_sigma_min: float
   :param dt_sigma_min: minimum travel time error allowed
   :type dlna_sigma_min: float
   :param dlna_sigma_min: minimum amplitude error allowed


.. py:function:: xcorr_shift(d, s)

   Determine the required time shift for peak cross-correlation of two arrays

   :type d: np.array
   :param d: observed time series array
   :type s:  np.array
   :param s: synthetic time series array


.. py:function:: subsample_xcorr_shift(d, s)

   Calculate the correlation time shift around the maximum amplitude of the
   synthetic trace `s` with subsample accuracy.

   :type d: obspy.core.trace.Trace
   :param d: observed waveform to calculate adjoint source
   :type s:  obspy.core.trace.Trace
   :param s: synthetic waveform to calculate adjoint source


