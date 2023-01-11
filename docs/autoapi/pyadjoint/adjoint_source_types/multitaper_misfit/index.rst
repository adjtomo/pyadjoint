:py:mod:`pyadjoint.adjoint_source_types.multitaper_misfit`
==========================================================

.. py:module:: pyadjoint.adjoint_source_types.multitaper_misfit

.. autoapi-nested-parse::

   Multitaper based phase and amplitude misfit and adjoint source.

   :authors:
       adjTomo Dev Team (adjtomo@gmail.com), 2022
       Youyi Ruan (youyir@princeton.edu), 2016
       Matthieu Lefebvre (ml15@princeton.edu), 2016
       Yanhua O. Yuan (yanhuay@princeton.edu), 2015
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pyadjoint.adjoint_source_types.multitaper_misfit.MultitaperMisfit



Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.adjoint_source_types.multitaper_misfit.calculate_adjoint_source



.. py:class:: MultitaperMisfit(observed, synthetic, config, windows, observed_2=None, synthetic_2=None, windows_2=None)

   A class to house the machinery of the multitaper misfit calculation. This is
   done with a class rather than a function to reduce the amount of unnecessary
   parameter passing between functions.

   .. py:method:: calculate_adjoint_source()

      Main processing function to calculate adjoint source for MTM.
      The logic here is that the function will perform a series of checks/
      calculations to check if MTM is valid for each given window. If
      any check/calculation fails, it will fall back to CCTM misfit for the
      given window.


   .. py:method:: calculate_dd_adjoint_source()

      Process double difference adjoint source. Requires second set of
      waveforms and windows.

      .. note::

          amplitude measurement stuff has been mostly left in the function
          (i.e., commented out) even it is not used, so that hopefully it is
          easier for someone in the future to implement it if they want.

      :rtype: (float, np.array, np.array, dict)
      :return: (misfit_sum_p, fp, fp_2, win_stats) == (
          total phase misfit  for the measurement,
          adjoint source for first data-synthetic pair,
          adjoint source for second data-synthetic pair,
          measurement information dictionary
          )


   .. py:method:: calculate_mt_adjsrc(s, tapers, nfreq_min, nfreq_max, dtau_mtm, dlna_mtm, wp_w, wq_w)

      Calculate the adjoint source for a multitaper measurement, which
      tapers synthetics in various windowed frequency-dependent tapers and
      scales them by phase dependent travel time measurements (which
      incorporate the observed data).

      :type s: np.array
      :param s: synthetic data array
      :type tapers: np.array
      :param tapers: array of DPPS windows shaped (num_taper, nlen_w)
      :type nfreq_min: int
      :param nfreq_min: minimum frequency for suitable MTM measurement
      :type nfreq_max: int
      :param nfreq_max: maximum frequency for suitable MTM measurement
      :type dtau_mtm: np.array
      :param dtau_mtm: phase dependent travel time measurements from mtm
      :type dlna_mtm: np.array
      :param dlna_mtm: phase dependent amplitude anomaly
      :type wp_w: np.array
      :param wp_w: phase-misfit error weighted frequency domain taper
      :type wq_w: np.array
      :param wq_w: amplitude-misfit error weighted frequency domain taper


   .. py:method:: calculate_dd_mt_adjsrc(s, s_2, tapers, nfreq_min, nfreq_max, df, dtau_mtm, dlna_mtm, wp_w, wq_w)

      Calculate the double difference adjoint source for multitaper
      measurement. Almost the same as `calculate_mt_adjsrc` but only addresses
      phase misfit and requres a second set of synthetics `s_2` which is
      processed in the same way as the first set `s`

      :type s: np.array
      :param s: synthetic data array
      :type s_2: np.array
      :param s_2: optional 2nd set of synthetics for double difference
          measurements only. This will change the output
      :type tapers: np.array
      :param tapers: array of DPPS windows shaped (num_taper, nlen_w)
      :type nfreq_min: int
      :param nfreq_min: minimum frequency for suitable MTM measurement
      :type nfreq_max: int
      :param nfreq_max: maximum frequency for suitable MTM measurement
      :type df: floats
      :param df: step length of frequency bins for FFT
      :type dtau_mtm: np.array
      :param dtau_mtm: phase dependent travel time measurements from mtm
      :type dlna_mtm: np.array
      :param dlna_mtm: phase dependent amplitude anomaly
      :type wp_w: np.array
      :param wp_w: phase-misfit error weighted frequency domain taper
      :type wq_w: np.array
      :param wq_w: amplitude-misfit error weighted frequency domain taper


   .. py:method:: calculate_freq_domain_taper(nfreq_min, nfreq_max, df, dtau_mtm, dlna_mtm, err_dt_cc, err_dlna_cc, err_dtau_mt, err_dlna_mt)

      Calculates frequency domain taper weighted by misfit (either CC or MTM)

      .. note::

          Frequency-domain tapers are based on adjusted frequency band and
          error estimation. They are not one of the filtering processes that
          needs to be applied to the adjoint source but rather a frequency
          domain weighting function for adjoint source and misfit function.

      :type nfreq_min: int
      :param nfreq_min: minimum frequency for suitable MTM measurement
      :type nfreq_max: int
      :param nfreq_max: maximum frequency for suitable MTM measurement
      :type df: floats
      :param df: step length of frequency bins for FFT
      :type dtau_mtm: np.array
      :param dtau_mtm: phase dependent travel time measurements from mtm
      :type dlna_mtm: np.array
      :param dlna_mtm: phase dependent amplitude anomaly
      :type err_dt_cc: float
      :param err_dt_cc: cross correlation time shift error
      :type err_dlna_cc: float
      :param err_dlna_cc: cross correlation amplitude anomaly error
      :type err_dtau_mt: np.array
      :param err_dtau_mt: phase-dependent timeshift error
      :type err_dlna_mt: np.array
      :param err_dlna_mt: phase-dependent amplitude error


   .. py:method:: calculate_multitaper(d, s, tapers, wvec, nfreq_min, nfreq_max, cc_tshift, cc_dlna)

      Measure phase-dependent time shifts and amplitude anomalies using
      the multitaper method

      .. note::
          Formerly `mt_measure`. Renamed for additional clarity and to match
          the CCTM function names

      :type d: np.array
      :param d: observed data array
      :type s: np.array
      :param s: synthetic data array
      :type tapers: np.array
      :param tapers: array of DPPS windows shaped (num_taper, nlen_w)
      :type wvec: np.array
      :param wvec: angular frequency array generated from Discrete Fourier
          Transform sample frequencies
      :type nfreq_min: int
      :param nfreq_min: minimum frequency for suitable MTM measurement
      :type nfreq_max: int
      :param nfreq_max: maximum frequency for suitable MTM measurement
      :type cc_tshift: float
      :param cc_tshift: cross correlation time shift
      :type cc_dlna: float
      :param cc_dlna: amplitude anomaly from cross correlation
      :rtype: tuple of np.array
      :return: (phi_w, abs_w, dtau_w, dlna_w);
          (frequency dependent phase anomaly,
          phase dependent amplitude anomaly,
          phase dependent cross-correlation time shift,
          phase dependent cross-correlation amplitude anomaly)


   .. py:method:: calculate_mt_error(d, s, tapers, wvec, nfreq_min, nfreq_max, cc_tshift, cc_dlna, phi_mtm, abs_mtm, dtau_mtm, dlna_mtm)

      Calculate multitaper error with Jackknife MT estimates.

      The jackknife estimator of a parameter is found by systematically
      leaving out each observation from a dataset and calculating the
      parameter estimate over the remaining observations and then aggregating
      these calculations.

      :type d: np.array
      :param d: observed data array
      :type s: np.array
      :param s: synthetic data array
      :type tapers: np.array
      :param tapers: array of DPPS windows shaped (num_taper, nlen_w)
      :type wvec: np.array
      :param wvec: angular frequency array generated from Discrete Fourier
          Transform sample frequencies
      :type nfreq_min: int
      :param nfreq_min: minimum frequency for suitable MTM measurement
      :type nfreq_max: int
      :param nfreq_max: maximum frequency for suitable MTM measurement
      :type cc_tshift: float
      :param cc_tshift: cross correlation time shift
      :type cc_dlna: float
      :param cc_dlna: amplitude anomaly from cross correlation
      :type phi_mtm: np.array
      :param phi_mtm: frequency dependent phase anomaly
      :type abs_mtm: np.array
      :param abs_mtm: phase dependent amplitude anomaly
      :type dtau_mtm: np.array
      :param dtau_mtm:  phase dependent cross-correlation time shift
      :type dlna_mtm: np.array
      :param dlna_mtm: phase dependent cross-correlation amplitude anomaly)
      :rtype: tuple of np.array
      :return: (err_phi, err_abs, err_dtau, err_dlna),
          (error in frequency dependent phase anomaly,
          error in phase dependent amplitude anomaly,
          error in phase dependent cross-correlation time shift,
          error in phase dependent cross-correlation amplitude anomaly)


   .. py:method:: calculate_freq_limits(df)

      Determine if a given window is suitable for multitaper measurements.
      If so, finds the maximum frequency range for the measurement using a
      spectrum of tapered synthetic waveforms

      First check if the window is suitable for mtm measurements, then
      find the maximum frequency point for measurement using the spectrum of
      tapered synthetics.

      .. note::
          formerly `frequency_limit`. renamed to be more descriptive. also
          split off earlier cycle check from this function into
          `check_sufficient_number_of_wavelengths`

      :type df: float
      :param df: step length of frequency bins for FFT
      :rtype: tuple
      :return (float, float, bool);
          (minimumum frequency, maximum frequency, continue with MTM?)


   .. py:method:: prepare_data_for_mtm(d, tshift, dlna, window)

      Re-window observed data to center on the optimal time shift, and
      scale by amplitude anomaly to get best matching waveforms for MTM

      :return:


   .. py:method:: check_time_series_acceptability(cc_tshift, nlen_w)

      Checking acceptability of the time series characteristics for MTM

      :type cc_tshift: float
      :param cc_tshift: time shift in unit [s]
      :type nlen_w: int
      :param nlen_w: window length in samples
      :rtype: bool
      :return: True if time series OK for MTM, False if fall back to CC


   .. py:method:: check_mtm_time_shift_acceptability(nfreq_min, nfreq_max, df, cc_tshift, dtau_mtm, sigma_dtau_mt)

      Check MTM time shift measurements to see if they are within allowable
      bounds set by the config. If any of the phases used in MTM do not
      meet criteria, we will fall back to CC measurement.

      .. note::
          formerly `mt_measure_select`, renamed for clarity

      :type nfreq_max: int
      :param nfreq_max: maximum in frequency domain
      :type nfreq_min: int
      :param nfreq_min: minimum in frequency domain
      :type df: floats
      :param df: step length of frequency bins for FFT
      :type cc_tshift: float
      :param cc_tshift: c.c. time shift
      :type dtau_mtm: np.array
      :param dtau_mtm: phase dependent travel time measurements from mtm
      :type sigma_dtau_mt: np.array
      :param sigma_dtau_mt: phase-dependent error of multitaper measurement
      :rtype: bool
      :return: flag for whether any of the MTM phases failed check


   .. py:method:: rewindow(data, left_sample, right_sample, shift)

      Align data in a window according to a given time shift. Will not fully
      shift if shifted window hits bounds of the data array

      :type data: np.array
      :param data: full data array to cut with shifted window
      :type left_sample: int
      :param left_sample: left window border
      :type right_sample: int
      :param right_sample: right window border
      :type shift: int
      :param shift: overall time shift in units of samples


   .. py:method:: _search_frequency_limit(is_search, index, nfreq_limit, spectra, water_threshold, c=10)
      :staticmethod:

      Search valid frequency range of spectra. If the spectra larger than
      10 * `water_threshold` it will trigger the search again, works like the
      heating thermostat.

      :type is_search: bool
      :param is_search: Logic switch
      :type index: int
      :param index: index of spectra
      :type nfreq_limit: int
      :param nfreq_limit: index of freqency limit searched
      :type spectra: int
      :param spectra: spectra of signal
      :type water_threshold: float
      :param water_threshold: optional triggering value to stop the search,
          if not given, defaults to Config value
      :type c: int
      :param c: constant scaling factor for water threshold.



.. py:function:: calculate_adjoint_source(observed, synthetic, config, windows, observed_2=None, synthetic_2=None, windows_2=None)

   Convenience wrapper function for MTM class to match the expected format
   of Pyadjoint. Contains the logic for what to return to User.

   :type observed: obspy.core.trace.Trace
   :param observed: observed waveform to calculate adjoint source
   :type synthetic:  obspy.core.trace.Trace
   :param synthetic: synthetic waveform to calculate adjoint source
   :type config: pyadjoint.config.ConfigCCTraveltime
   :param config: Config class with parameters to control processing
   :type windows: list of tuples
   :param windows: [(left, right),...] representing left and right window
       borders to be used to calculate misfit and adjoint sources
   :type observed_2: obspy.core.trace.Trace
   :param observed_2: second observed waveform to calculate adjoint sources
       from station pairs
   :type synthetic_2:  obspy.core.trace.Trace
   :param synthetic_2: second synthetic waveform to calculate adjoint sources
       from station pairs
   :type windows_2: list of tuples
   :param windows_2: [(left, right),...] representing left and right window
       borders to be tapered in units of seconds since first sample in data
       array. Used to window `observed_2` and `synthetic_2`


