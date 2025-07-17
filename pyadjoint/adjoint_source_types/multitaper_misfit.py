#!/usr/bin/env python3
"""
Multitaper based phase and amplitude misfit and adjoint source.

:authors:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Youyi Ruan (youyir@princeton.edu), 2016
    Matthieu Lefebvre (ml15@princeton.edu), 2016
    Yanhua O. Yuan (yanhuay@princeton.edu), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
from scipy.integrate import simpson

from pyadjoint import logger
from pyadjoint.utils.dpss import dpss_windows
from pyadjoint.utils.cctm import (calculate_cc_shift, calculate_cc_adjsrc,
                                  calculate_dd_cc_shift, calculate_dd_cc_adjsrc,
                                  )
from pyadjoint.utils.signal import (window_taper, get_window_info,
                                    process_cycle_skipping)


class MultitaperMisfit:
    """
    A class to house the machinery of the multitaper misfit calculation. This is
    done with a class rather than a function to reduce the amount of unnecessary
    parameter passing between functions.
    """
    def __init__(self, observed, synthetic, config, windows,
                 observed_2=None, synthetic_2=None, windows_2=None
                 ):
        """
        Initialize Multitaper Misfit adjoint source creator

        :type observed: obspy.core.trace.Trace
        :param observed: observed waveform to calculate adjoint source
        :type synthetic:  obspy.core.trace.Trace
        :param synthetic: synthetic waveform to calculate adjoint source
        :type config: pyadjoint.config.Multitaper
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
        """
        assert (config.__class__.__name__ == "ConfigMultitaper"), \
            "Incorrect configuration class passed to CCTraveltime misfit"

        self.observed = observed
        self.synthetic = synthetic
        self.config = config
        self.windows = windows

        # For optional double-difference measurements
        self.observed_2 = observed_2
        self.synthetic_2 = synthetic_2
        self.windows_2 = windows_2

        # Calculate some information to be used for MTM measurements
        # Assumed that second set of waveforms (if provided) have same qualities
        self.nlen_f = 2 ** self.config.lnpt
        self.nlen_data = len(synthetic.data)  # length in samples
        self.dt = synthetic.stats.delta  # sampling rate
        self.tlen_data = self.nlen_data * self.dt  # length in time [s]

    def calculate_adjoint_source(self):
        """
        Main processing function to calculate adjoint source for MTM.
        The logic here is that the function will perform a series of checks/
        calculations to check if MTM is valid for each given window. If
        any check/calculation fails, it will fall back to CCTM misfit for the
        given window.
        """
        # Arrays for adjoint sources w.r.t time shift (p) and amplitude (q)
        fp = np.zeros(self.nlen_data)
        fq = np.zeros(self.nlen_data)

        misfit_sum_p = 0.0
        misfit_sum_q = 0.0
        win_stats = []

        # Loop over time windows and calculate misfit for each window range
        for window in self.windows:
            # `is_mtm` determines whether we use MTM (T) or CC (F) for misfit
            is_mtm = True

            left_sample, right_sample, nlen_w = get_window_info(window, self.dt)
            fp_t = np.zeros(nlen_w)
            fq_t = np.zeros(nlen_w)
            misfit_p = 0.
            misfit_q = 0.

            # Pre-allocate arrays for memory efficiency
            d = np.zeros(nlen_w)
            s = np.zeros(nlen_w)

            # d and s represent the windowed data and synthetic arrays
            d[0: nlen_w] = self.observed.data[left_sample: right_sample]
            s[0: nlen_w] = self.synthetic.data[left_sample: right_sample]

            # Taper windowed signals in place
            window_taper(d, taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)
            window_taper(s, taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)

            # Calculate cross correlation time shift, amplitude anomaly and
            # errors. Config passed as **kwargs to control constants required
            # by function
            cc_tshift, cc_dlna, sigma_dt_cc, sigma_dlna_cc = calculate_cc_shift(
                d=d, s=s, dt=self.dt, **vars(self.config)
            )

            # Perform a series of checks to see if MTM is valid for the data
            # This will only loop once, but allows us to break if a check fail
            while is_mtm is True:
                is_mtm = self.check_time_series_acceptability(
                        cc_tshift=cc_tshift, nlen_w=nlen_w) 
                if is_mtm is False:
                    break

                # Shift and scale observed data 'd' to match synthetics, make
                # sure the time shift doesn't go passed time series' bounds
                d, is_mtm = self.prepare_data_for_mtm(d=d, tshift=cc_tshift,
                                                      dlna=cc_dlna,
                                                      window=window)
                if is_mtm is False:
                    logger.info(f"reject MTM: adjusted CC shift: {cc_tshift} is"
                                f"out of bounds of time series")
                    logger.debug(f"win = [{left_sample * self.dt}, "
                                 f"{right_sample * self.dt}]")
                    break

                # Determine FFT information related to frequency bands
                # TODO: Sampling rate was set to observed delta, is dt the same?
                freq = np.fft.fftfreq(n=self.nlen_f, d=self.dt)
                df = freq[1] - freq[0]  # delta_f: frequency step
                wvec = freq * 2 * np.pi  # omega vector: angular frequency
                # dw = wvec[1] - wvec[0]  # TODO: check to see if dw is not used
                logger.debug("delta_f (frequency sampling) = {df}")

                # Check for sufficient frequency range given taper bandwith
                nfreq_min, nfreq_max, is_mtm = self.calculate_freq_limits(df)
                if is_mtm is False:
                    logger.info("reject MTM: frequency range narrower than "
                                "half taper bandwith")
                    break

                # Determine taper bandwith in frequency domain
                tapert, eigens = dpss_windows(
                    n=nlen_w, half_nbw=self.config.mt_nw,
                    k_max=self.config.num_taper, low_bias=False
                )
                is_mtm = np.isfinite(eigens).all()
                if is_mtm is False:
                    logger.warning("reject MTM: error constructing DPSS tapers")
                    logger.debug(f"eigenvalues: {eigens}")
                    break

                # Check if tapers are properly generated. In rare cases
                # (e.g., [nw=2.5, nlen=61] or [nw=4.0, nlen=15]) certain
                # eigenvalues can not be found and associated eigentaper is NaN
                tapers = tapert.T * np.sqrt(nlen_w)
                phi_mtm, abs_mtm, dtau_mtm, dlna_mtm = \
                    self.calculate_multitaper(
                        d=d, s=s, tapers=tapers, wvec=wvec, nfreq_min=nfreq_min,
                        nfreq_max=nfreq_max, cc_tshift=cc_tshift,
                        cc_dlna=cc_dlna
                    )

                # Calculate multi-taper error estimation if requested
                if self.config.use_mt_error:
                    sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, \
                        sigma_dlna_mt = self.calculate_mt_error(
                            d=d, s=s, tapers=tapers, wvec=wvec,
                            nfreq_min=nfreq_min, nfreq_max=nfreq_max,
                            cc_tshift=cc_tshift, cc_dlna=cc_dlna,
                            phi_mtm=phi_mtm, abs_mtm=abs_mtm,
                            dtau_mtm=dtau_mtm, dlna_mtm=dlna_mtm)
                else:
                    sigma_dtau_mt = np.zeros(self.nlen_f)
                    sigma_dlna_mt = np.zeros(self.nlen_f)

                # Check if the multitaper measurements fail selection criteria
                is_mtm = self.check_mtm_time_shift_acceptability(
                                nfreq_min=nfreq_min, nfreq_max=nfreq_max, df=df,
                                cc_tshift=cc_tshift, dtau_mtm=dtau_mtm,
                                sigma_dtau_mt=sigma_dtau_mt)
                if is_mtm is False:
                    break

                # We made it! If the loop is still running after this point,
                # then we will use MTM for adjoint source calculation

                # Frequency domain taper weighted by measurement error
                wp_w, wq_w = self.calculate_freq_domain_taper(
                    nfreq_min=nfreq_min, nfreq_max=nfreq_max, df=df,
                    dtau_mtm=dtau_mtm, dlna_mtm=dlna_mtm, err_dt_cc=sigma_dt_cc,
                    err_dlna_cc=sigma_dlna_cc, err_dtau_mt=sigma_dtau_mt,
                    err_dlna_mt=sigma_dlna_mt,
                )

                # Misfit is defined as the error-weighted measurements
                dtau_mtm_weigh_sqr = dtau_mtm ** 2 * wp_w
                dlna_mtm_weigh_sqr = dlna_mtm ** 2 * wq_w
                misfit_p = 0.5 * 2.0 * simpson(y=dtau_mtm_weigh_sqr, dx=df)
                misfit_q = 0.5 * 2.0 * simpson(y=dlna_mtm_weigh_sqr, dx=df)

                logger.info("calculating misfit and adjoint source with MTM")
                fp_t, fq_t = self.calculate_mt_adjsrc(
                    s=s, tapers=tapers,  nfreq_min=nfreq_min,
                    nfreq_max=nfreq_max, dtau_mtm=dtau_mtm, dlna_mtm=dlna_mtm,
                    wp_w=wp_w, wq_w=wq_w
                )
                win_stats.append(
                    {"left": left_sample * self.dt,
                     "right": right_sample * self.dt,
                     "type": "multitaper",
                     "measurement_type": self.config.measure_type,
                     "misfit_dt": misfit_p,
                     "misfit_dlna": misfit_q,
                     "sigma_dt": sigma_dt_cc,
                     "sigma_dlna": sigma_dlna_cc,
                     "tshift": np.mean(dtau_mtm[nfreq_min:nfreq_max]),
                     "dlna": np.mean(dlna_mtm[nfreq_min:nfreq_max]),
                     }
                )
                break
            # If at some point MTM broke out of the loop, this code block will
            # execute and calculate a CC adjoint source and misfit instead
            if is_mtm is False:
                logger.info("calculating misfit and adjoint source with CCTM")
                misfit_p, misfit_q, fp_t, fq_t = \
                    calculate_cc_adjsrc(s=s, tshift=cc_tshift, dlna=cc_dlna,
                                        dt=self.dt, sigma_dt=sigma_dt_cc,
                                        sigma_dlna=sigma_dlna_cc)
                win_stats.append(
                    {"left": left_sample * self.dt,
                     "right": right_sample * self.dt,
                     "type": "cross_correlation_traveltime",
                     "measurement_type": self.config.measure_type,
                     "misfit_dt": misfit_p,
                     "misfit_dlna": misfit_q,
                     "sigma_dt": sigma_dt_cc,
                     "sigma_dlna": sigma_dlna_cc,
                     "dt": cc_tshift,
                     "dlna": cc_dlna,
                     }
                )

            # Taper windowed adjoint source before including in final array
            window_taper(fp_t[0:nlen_w],
                         taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)
            window_taper(fq_t[0:nlen_w],
                         taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)

            # Place windowed adjoint source within the correct time location
            # w.r.t the entire synthetic seismogram
            fp_wind = np.zeros(len(self.synthetic.data))
            fq_wind = np.zeros(len(self.synthetic.data))
            fp_wind[left_sample: right_sample] = fp_t[0:nlen_w]
            fq_wind[left_sample: right_sample] = fq_t[0:nlen_w]

            # Add the windowed adjoint source to the full adjoint source
            fp += fp_wind
            fq += fq_wind

            # Increment total misfit value by misfit of windows
            misfit_sum_p += misfit_p
            misfit_sum_q += misfit_q

        return misfit_sum_p, misfit_sum_q, fp, fq, win_stats

    def calculate_dd_adjoint_source(self):
        """
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
        """
        # Arrays for adjoint sources w.r.t time shift (p) and amplitude (q)
        fp = np.zeros(self.nlen_data)
        fp_2 = np.zeros(self.nlen_data)

        misfit_sum_p = 0.0
        # misfit_sum_q = 0.0
        win_stats = []

        # Loop over time windows and calculate misfit for each window range
        for window, window_2 in zip(self.windows, self.windows_2):
            # `is_mtm` determines whether we use MTM (T) or CC (F) for misfit
            is_mtm = True

            # Prepare first set of waveforms
            left_sample, right_sample, nlen_w = get_window_info(window, self.dt)
            fp_t = np.zeros(nlen_w)
            misfit_p = 0.
            # misfit_q = 0.
            # Pre-allocate arrays for memory efficiency
            d = np.zeros(nlen_w)
            s = np.zeros(nlen_w)
            # d and s represent the windowed data and synthetic arrays
            d[0: nlen_w] = self.observed.data[left_sample: right_sample]
            s[0: nlen_w] = self.synthetic.data[left_sample: right_sample]

            # Prepare second set of waveforms
            left_sample_2, right_sample_2, nlen_w_2 = \
                get_window_info(window_2, self.dt)
            fp_2_t = np.zeros(nlen_w_2)

            # Pre-allocate arrays for memory efficiency
            d_2 = np.zeros(nlen_w)
            s_2 = np.zeros(nlen_w)
            # d and s represent the windowed data and synthetic arrays
            d_2[0: nlen_w_2] = \
                self.observed_2.data[left_sample_2: right_sample_2]
            s_2[0: nlen_w_2] = \
                self.synthetic_2.data[left_sample_2: right_sample_2]

            # Taper windowed signals (modify arrays in place)
            for arr in [d, s, d_2, s_2]:
                window_taper(arr, taper_percentage=self.config.taper_percentage,
                             taper_type=self.config.taper_type)

            # Calculate double difference cross correlation time shift for
            # both sets of waveforms
            cc_tshift, cc_tshift_obs, cc_tshift_syn, cc_dlna_obs, cc_dlna_syn, \
                sigma_dt_cc, sigma_dlna_cc = calculate_dd_cc_shift(
                    d=d, s=s, d_2=d_2, s_2=s_2, dt=self.dt, **vars(self.config)
                    )

            # Perform a series of checks to see if MTM is valid for the data
            # This will only loop once, but allows us to break if a check fails
            while is_mtm is True:
                is_mtm = self.check_time_series_acceptability(
                        cc_tshift=cc_tshift, nlen_w=nlen_w) 
                if is_mtm is False:
                    break

                # Shift and scale observed data 'd' to match second set: `d_2`
                d, is_mtm_obs = self.prepare_data_for_mtm(
                    d=d, tshift=cc_tshift_obs, dlna=cc_dlna_obs, window=window
                )
                # Shift and scale synthetics 's' to match second set: `s_2`
                s, is_mtm_syn = self.prepare_data_for_mtm(
                    d=s, tshift=cc_tshift_syn, dlna=cc_dlna_syn, window=window
                )
                if is_mtm_obs is False or is_mtm_syn is False:
                    logger.info(f"reject MTM: adjusted CC shift: {cc_tshift} is"
                                f"out of bounds of time series")
                    logger.debug(f"win = [{left_sample * self.dt}, "
                                 f"{right_sample * self.dt}]")
                    break

                # Determine FFT information related to frequency bands
                # TODO: Sampling rate was set to observed delta, is dt the same?
                freq = np.fft.fftfreq(n=self.nlen_f, d=self.dt)
                df = freq[1] - freq[0]  # delta_f: frequency step
                wvec = freq * 2 * np.pi  # omega vector: angular frequency
                logger.debug("delta_f (frequency sampling) = {df}")

                # Check for sufficient frequency range given taper bandwith
                nfreq_min, nfreq_max, is_mtm = self.calculate_freq_limits(df)
                if is_mtm is False:
                    logger.info("reject MTM: frequency range narrower than "
                                "half taper bandwith")
                    break

                # Determine taper bandwith in frequency domain
                tapert, eigens = dpss_windows(
                    n=nlen_w, half_nbw=self.config.mt_nw,
                    k_max=self.config.num_taper, low_bias=False
                )
                is_mtm = np.isfinite(eigens).all()
                if is_mtm is False:
                    logger.warning("reject MTM: error constructing DPSS tapers")
                    logger.debug(f"eigenvalues: {eigens}")
                    break

                # Check if tapers are properly generated. In rare cases
                # (e.g., [nw=2.5, nlen=61] or [nw=4.0, nlen=15]) certain
                # eigenvalues can not be found and associated eigentaper is NaN
                tapers = tapert.T * np.sqrt(nlen_w)
                phi_mtm_obs, abs_mtm_obs, dtau_mtm_obs, dlna_mtm_obs = \
                    self.calculate_multitaper(
                        d=d, s=d_2, tapers=tapers, wvec=wvec,
                        nfreq_min=nfreq_min, nfreq_max=nfreq_max,
                        cc_tshift=cc_tshift_obs, cc_dlna=cc_dlna_obs
                    )
                phi_mtm_syn, abs_mtm_syn, dtau_mtm_syn, dlna_mtm_syn = \
                    self.calculate_multitaper(
                        d=s, s=s_2, tapers=tapers, wvec=wvec,
                        nfreq_min=nfreq_min, nfreq_max=nfreq_max,
                        cc_tshift=cc_tshift_syn, cc_dlna=cc_dlna_syn
                    )

                # Measurements are difference between double differences
                # phi_mtm = phi_mtm_syn - phi_mtm_obs
                # abs_mtm = abs_mtm_syn - abs_mtm_obs
                dtau_mtm = dtau_mtm_syn - dtau_mtm_obs
                dlna_mtm = dlna_mtm_syn - dlna_mtm_obs

                # Calculate multi-taper error estimation if requested
                # FIXME: should dtau_mtm and dlna_mtm be the 'obs' or 'diff' v.?
                if self.config.use_mt_error:
                    sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, \
                        sigma_dlna_mt = self.calculate_mt_error(
                            d=d, s=s, tapers=tapers, wvec=wvec,
                            nfreq_min=nfreq_min, nfreq_max=nfreq_max,
                            cc_tshift=cc_tshift, cc_dlna=cc_dlna_obs,
                            phi_mtm=phi_mtm_obs, abs_mtm=abs_mtm_obs,
                            # dtau_mtm=dtau_mtm_obs, dlna_mtm=dlna_mtm_obs # ?
                            dtau_mtm=dtau_mtm, dlna_mtm=dlna_mtm
                            )
                else:
                    sigma_dtau_mt = np.zeros(self.nlen_f)
                    sigma_dlna_mt = np.zeros(self.nlen_f)

                # Check if the multitaper measurements fail selection criteria
                is_mtm = self.check_mtm_time_shift_acceptability(
                                nfreq_min=nfreq_min, nfreq_max=nfreq_max, df=df,
                                cc_tshift=cc_tshift, dtau_mtm=dtau_mtm,
                                sigma_dtau_mt=sigma_dtau_mt)
                if is_mtm is False:
                    break

                # We made it! If the loop is still running after this point,
                # then we will use MTM for adjoint source calculation

                # Frequency domain taper weighted by measurement error
                wp_w, wq_w = self.calculate_freq_domain_taper(
                    nfreq_min=nfreq_min, nfreq_max=nfreq_max, df=df,
                    dtau_mtm=dtau_mtm, dlna_mtm=dlna_mtm, err_dt_cc=sigma_dt_cc,
                    err_dlna_cc=sigma_dlna_cc, err_dtau_mt=sigma_dtau_mt,
                    err_dlna_mt=sigma_dlna_mt,
                )

                # Misfit is defined as the error-weighted measurements
                # TODO dlna misfit not calculated, only phase (dtau)
                dtau_mtm_weigh_sqr = dtau_mtm ** 2 * wp_w
                misfit_p = 0.5 * 2.0 * simpson(y=dtau_mtm_weigh_sqr, dx=df)

                logger.info("calculate double difference adjoint source w/ MTM")
                fp_t, fp_2_t = self.calculate_dd_mt_adjsrc(
                    s=s, s_2=s_2, tapers=tapers,  nfreq_min=nfreq_min,
                    nfreq_max=nfreq_max, df=df, dtau_mtm=dtau_mtm, 
                    dlna_mtm=dlna_mtm, wp_w=wp_w, wq_w=wq_w
                )

                win_stats.append(
                    {"left": left_sample * self.dt,
                     "right": right_sample * self.dt,
                     "type": "dd_multitaper",
                     "measurement_type": self.config.measure_type,
                     "misfit_dt": misfit_p,
                     "sigma_dt": sigma_dt_cc,
                     "sigma_dlna": sigma_dlna_cc,
                     "tshift": np.mean(dtau_mtm[nfreq_min:nfreq_max]),
                     "dlna": np.mean(dlna_mtm[nfreq_min:nfreq_max]),
                     }
                )
                break

            # If at some point MTM broke out of the loop, this code block will
            # execute and calculate a CC adjoint source and misfit instead
            if is_mtm is False:
                logger.info("calculating adjoint source with double diff. CCTM")
                misfit_p, misfit_q, fp_t, fp_2_t, fq_t, fq_2_t = \
                    calculate_dd_cc_adjsrc(s=s, s_2=s_2, tshift=cc_tshift,
                                           dlna=cc_dlna_obs, dt=self.dt,
                                           sigma_dt=sigma_dt_cc,
                                           sigma_dlna=sigma_dlna_cc)
                win_stats.append(
                    {"left": left_sample * self.dt,
                     "right": right_sample * self.dt,
                     "type": "dd_cross_correlation_traveltime",
                     "measurement_type": self.config.measure_type,
                     "misfit_dt": misfit_p,
                     "misfit_dlna": misfit_q,
                     "sigma_dt": sigma_dt_cc,
                     "sigma_dlna": sigma_dlna_cc,
                     "dt": cc_tshift,
                     "dlna": cc_dlna_obs,
                     }
                )

            # Taper windowed adjoint source before including in final array
            window_taper(fp_t[0:nlen_w],
                         taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)
            window_taper(fp_2_t[0:nlen_w],
                         taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)

            # Place windowed adjoint source within the correct time location
            # w.r.t the entire synthetic seismogram
            fp_wind = np.zeros(len(self.synthetic.data))
            fp_2_wind = np.zeros(len(self.synthetic.data))
            fp_wind[left_sample: right_sample] = fp_t[0:nlen_w]
            fp_2_wind[left_sample: right_sample] = fp_2_t[0:nlen_w]

            # Add the windowed adjoint source to the full adjoint source
            fp += fp_wind
            fp_2 += fp_2_wind

            # Increment total misfit value by misfit of windows
            misfit_sum_p += misfit_p

        return misfit_sum_p, fp, fp_2, win_stats

    def calculate_mt_adjsrc(self, s, tapers, nfreq_min, nfreq_max,
                            dtau_mtm, dlna_mtm, wp_w, wq_w):
        """
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
        """
        nlen_t = len(s)
        ntaper = len(tapers[0])

        # Start pieceing together transfer functions that will be applie to
        # the synthetics
        bottom_p = np.zeros(self.nlen_f, dtype=complex)
        bottom_q = np.zeros(self.nlen_f, dtype=complex)

        s_tw = np.zeros((self.nlen_f, ntaper), dtype=complex)
        s_tvw = np.zeros((self.nlen_f, ntaper), dtype=complex)

        # Construct the bottom term of the adjoint formula which requires
        # summed contributions from each of the taper bands
        for itaper in range(0, ntaper):
            taper = np.zeros(self.nlen_f)
            taper[0:nlen_t] = tapers[0:nlen_t, itaper]

            # Taper synthetics (s_t) and take the derivative (s_tv)
            s_t = s * taper[0:nlen_t]
            s_tv = np.gradient(s_t, self.dt)

            # Apply FFT to tapered measurements to get to freq. domain.
            s_tw[:, itaper] = np.fft.fft(s_t, self.nlen_f)[:] * self.dt
            s_tvw[:, itaper] = np.fft.fft(s_tv, self.nlen_f)[:] * self.dt

            # Calculate bottom term of the adjoint equation
            bottom_p[:] = (
                    bottom_p[:] +
                    s_tvw[:, itaper] * s_tvw[:, itaper].conjugate()
            )
            bottom_q[:] = (
                    bottom_q[:] +
                    s_tw[:, itaper] * s_tw[:, itaper].conjugate()
            )

        # Now we generate the adjoint sources using each of the tapers
        fp_t = np.zeros(nlen_t)
        fq_t = np.zeros(nlen_t)

        for itaper in range(0, ntaper):
            taper = np.zeros(self.nlen_f)
            taper[0: nlen_t] = tapers[0:nlen_t, itaper]

            # Calculate the full adjoint terms pj(w), qj(w)
            p_w = np.zeros(self.nlen_f, dtype=complex)
            q_w = np.zeros(self.nlen_f, dtype=complex)

            p_w[nfreq_min:nfreq_max] = (
                    s_tvw[nfreq_min:nfreq_max, itaper] /
                    bottom_p[nfreq_min:nfreq_max]
            )
            q_w[nfreq_min:nfreq_max] = (
                    -1 * s_tw[nfreq_min:nfreq_max, itaper] /
                    bottom_q[nfreq_min:nfreq_max]
            )

            # weight the adjoint terms by the phase + amplitude measurements
            p_w *= dtau_mtm * wp_w  # phase
            q_w *= dlna_mtm * wq_w  # amplitude

            # inverse FFT of weighted adjoint to get back to the time domain
            p_wt = np.fft.ifft(p_w, self.nlen_f).real * 2. / self.dt
            q_wt = np.fft.ifft(q_w, self.nlen_f).real * 2. / self.dt

            # Taper adjoint term before adding it back to full adj source
            fp_t[0:nlen_t] += p_wt[0:nlen_t] * taper[0:nlen_t]
            fq_t[0:nlen_t] += q_wt[0:nlen_t] * taper[0:nlen_t]

        return fp_t, fq_t

    def calculate_dd_mt_adjsrc(self, s, s_2, tapers, nfreq_min, nfreq_max, df,
                               dtau_mtm, dlna_mtm, wp_w, wq_w):
        """
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
        """
        nlen_t = len(s)
        ntaper = len(tapers[0])

        # Set up to piece together transfer functions that will be applied to
        # the synthetics. Sets up arrays for memory efficiency
        s_tw = np.zeros((self.nlen_f, ntaper), dtype=complex)
        s_tvw = np.zeros((self.nlen_f, ntaper), dtype=complex)

        s_2_tw = np.zeros((self.nlen_f, ntaper), dtype=complex)
        s_2_tvw = np.zeros((self.nlen_f, ntaper), dtype=complex)

        bottom_p = np.zeros(self.nlen_f, dtype=complex)
        bottom_p_2 = np.zeros(self.nlen_f, dtype=complex)

        # Construct the bottom term of the adjoint formula which requires
        # summed contributions from each of the taper bands
        for itaper in range(0, ntaper):
            taper = np.zeros(self.nlen_f)
            taper[0:nlen_t] = tapers[0:nlen_t, itaper]

            # Taper synthetics (s_t) and take the derivative (s_tv)
            s_t = s * taper[0:nlen_t]
            s_tv = np.gradient(s_t, self.dt)
            # Apply FFT to tapered measurements to get to freq. domain.
            s_tw[:, itaper] = np.fft.fft(s_t, self.nlen_f)[:] * self.dt
            s_tvw[:, itaper] = np.fft.fft(s_tv, self.nlen_f)[:] * self.dt

            # Perform same tasks but for second set synthetics
            s_2_t = s_2 * taper[0:nlen_t]
            s_2_tv = np.gradient(s_2_t, self.dt)
            s_2_tw[:, itaper] = np.fft.fft(s_2_t, self.nlen_f)[:] * self.dt
            s_2_tvw[:, itaper] = np.fft.fft(s_2_tv,
                                            self.nlen_f)[:] * self.dt

            # Calculate bottom term of the adjoint equation
            bottom_p[:] = (
                    bottom_p[:] +
                    s_tvw[:, itaper] * s_2_tvw[:, itaper].conjugate()
            )
            bottom_p_2[:] = (
                    bottom_p_2[:] +
                    s_2_tvw[:, itaper] * s_tvw[:, itaper].conjugate()
            )

        # Now we generate the adjoint sources using each of the tapers
        fp_t = np.zeros(nlen_t)
        fp_2_t = np.zeros(nlen_t)

        for itaper in range(0, ntaper):
            taper = np.zeros(self.nlen_f)
            taper[0: nlen_t] = tapers[0:nlen_t, itaper]

            # Calculate the full adjoint terms pj(w), qj(w)
            p_w = np.zeros(self.nlen_f, dtype=complex)
            p_2_w = np.zeros(self.nlen_f, dtype=complex)

            p_w[nfreq_min:nfreq_max] = (
                    -1. * s_2_tvw[nfreq_min:nfreq_max, itaper] /
                    bottom_p_2[nfreq_min:nfreq_max]
            )
            p_2_w[nfreq_min:nfreq_max] = (
                    1. * s_tvw[nfreq_min:nfreq_max, itaper] /
                    bottom_p[nfreq_min:nfreq_max]
            )

            # weight the adjoint terms by the phase measurements
            p_w *= dtau_mtm * wp_w  # phase
            p_2_w *= dtau_mtm * wq_w  # amplitude

            # inverse FFT of weighted adjoint to get back to the time domain
            p_wt = np.fft.ifft(p_w, self.nlen_f).real * 2. / self.dt
            p_2_wt = np.fft.ifft(p_2_w, self.nlen_f).real * 2. / self.dt

            # Taper adjoint term before adding it back to full adj source
            fp_t[0:nlen_t] += p_wt[0:nlen_t] * taper[0:nlen_t]
            fp_2_t[0:nlen_t] += p_2_wt[0:nlen_t] * taper[0:nlen_t]

        return fp_t, fp_2_t

    def calculate_freq_domain_taper(self, nfreq_min, nfreq_max, df,
                                    dtau_mtm, dlna_mtm, err_dt_cc,
                                    err_dlna_cc, err_dtau_mt, err_dlna_mt):
        """
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
        """
        w_taper = np.zeros(self.nlen_f)

        win_taper_len = nfreq_max - nfreq_min
        win_taper = np.ones(win_taper_len)

        # Createsa cosine taper over a range of frequencies in freq. domain
        window_taper(win_taper, taper_percentage=1.0, taper_type="cos_p10")
        w_taper[nfreq_min: nfreq_max] = win_taper[0:win_taper_len]

        # Normalization factor, factor 2 is needed for integration -inf to inf
        ffac = 2.0 * df * np.sum(w_taper[nfreq_min: nfreq_max])
        logger.debug(f"frequency bound (idx): [{nfreq_min}, {nfreq_max - 1}] "
                     f"(Hz) [{df * (nfreq_min - 1)}, {df * nfreq_max}]"
                     )
        logger.debug(f"frequency domain taper normalization coeff: {ffac}")
        logger.debug(f"frequency domain sampling length df={df}")
        if ffac <= 0.0:
            logger.warning("frequency band too narrow:")
            logger.warning(f"fmin={nfreq_min}, fmax={nfreq_max}, ffac={ffac}")

        # Normalized, tapered window in the frequency domain
        wp_w = w_taper / ffac
        wq_w = w_taper / ffac

        # Choose whether to scale by CC error or to by calculated MT errors
        if self.config.use_cc_error:
            wp_w /= err_dt_cc ** 2
            wq_w /= err_dlna_cc ** 2
        elif self.config.use_mt_error:
            dtau_wtr = (
                    self.config.water_threshold *
                    np.sum(np.abs(dtau_mtm[nfreq_min: nfreq_max])) /
                    (nfreq_max - nfreq_min)
            )
            dlna_wtr = (
                    self.config.water_threshold *
                    np.sum(np.abs(dlna_mtm[nfreq_min: nfreq_max])) /
                    (nfreq_max - nfreq_min)
            )

            err_dtau_mt[nfreq_min: nfreq_max] = \
                err_dtau_mt[nfreq_min: nfreq_max] + dtau_wtr * \
                (err_dtau_mt[nfreq_min: nfreq_max] < dtau_wtr)
            err_dlna_mt[nfreq_min: nfreq_max] = \
                err_dlna_mt[nfreq_min: nfreq_max] + dlna_wtr * \
                (err_dlna_mt[nfreq_min: nfreq_max] < dlna_wtr)

            wp_w[nfreq_min: nfreq_max] = (
                    wp_w[nfreq_min: nfreq_max] /
                    ((err_dtau_mt[nfreq_min: nfreq_max]) ** 2)
            )
            wq_w[nfreq_min: nfreq_max] = (
                    wq_w[nfreq_min: nfreq_max] /
                    ((err_dlna_mt[nfreq_min: nfreq_max]) ** 2)
            )

        return wp_w, wq_w

    def calculate_multitaper(self, d, s, tapers, wvec, nfreq_min, nfreq_max,
                             cc_tshift, cc_dlna):
        """
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
        """
        # Initialize some constants for convenience
        nlen_t = len(d)
        ntaper = len(tapers[0])
        fnum = int(self.nlen_f / 2 + 1)

        # Initialize empty arrays to be filled by FFT calculations
        top_tf = np.zeros(self.nlen_f, dtype=complex)
        bot_tf = np.zeros(self.nlen_f, dtype=complex)

        # Multitaper measurements
        for itaper in range(0, ntaper):
            taper = np.zeros(nlen_t)
            taper[0:nlen_t] = tapers[0:nlen_t, itaper]

            # Apply time-domain multi-tapered measurements
            d_t = np.zeros(nlen_t, dtype=complex)
            s_t = np.zeros(nlen_t, dtype=complex)

            d_t[0:nlen_t] = d[0:nlen_t] * taper[0:nlen_t]
            s_t[0:nlen_t] = s[0:nlen_t] * taper[0:nlen_t]

            d_tw = np.fft.fft(d_t, self.nlen_f) * self.dt
            s_tw = np.fft.fft(s_t, self.nlen_f) * self.dt

            # Calculate top and bottom of MT transfer function
            top_tf[:] = top_tf[:] + d_tw[:] * s_tw[:].conjugate()
            bot_tf[:] = bot_tf[:] + s_tw[:] * s_tw[:].conjugate()

        # Calculate water level for transfer function
        wtr_use = (max(abs(bot_tf[0:fnum])) *
                   self.config.transfunc_waterlevel ** 2)

        # Create transfer function
        trans_func = np.zeros(self.nlen_f, dtype=complex)
        for i in range(nfreq_min, nfreq_max):
            if abs(bot_tf[i]) < wtr_use:
                trans_func[i] = top_tf[i] / bot_tf[i]
            else:
                trans_func[i] = top_tf[i] / (bot_tf[i] + wtr_use)

        # Estimate phase and amplitude anomaly from transfer function
        phi_w = np.zeros(self.nlen_f)
        abs_w = np.zeros(self.nlen_f)
        dtau_w = np.zeros(self.nlen_f)
        dlna_w = np.zeros(self.nlen_f)

        # Calculate the phase anomaly
        phi_w[nfreq_min:nfreq_max] = np.arctan2(
            trans_func[nfreq_min:nfreq_max].imag,
            trans_func[nfreq_min:nfreq_max].real
        )
        phi_w = process_cycle_skipping(phi_w=phi_w, nfreq_max=nfreq_max,
                                       nfreq_min=nfreq_min, wvec=wvec,
                                       phase_step=self.config.phase_step)

        # Calculate amplitude anomaly
        abs_w[nfreq_min:nfreq_max] = np.abs(trans_func[nfreq_min:nfreq_max])

        # Add the CC measurements to the transfer function
        dtau_w[0] = cc_tshift
        dtau_w[max(nfreq_min, 1): nfreq_max] = \
            - 1.0 / wvec[max(nfreq_min, 1): nfreq_max] * \
            phi_w[max(nfreq_min, 1): nfreq_max] + cc_tshift

        dlna_w[nfreq_min:nfreq_max] = np.log(
            abs_w[nfreq_min:nfreq_max]) + cc_dlna

        return phi_w, abs_w, dtau_w, dlna_w

    def calculate_mt_error(self, d, s, tapers, wvec, nfreq_min, nfreq_max,
                           cc_tshift, cc_dlna, phi_mtm, abs_mtm, dtau_mtm,
                           dlna_mtm):
        """
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
        """
        nlen_t = len(d)
        ntaper = len(tapers[0])
        logger.debug("Number of tapers used: %d" % ntaper)

        # Jacknife MT estimates. Initialize arrays for memory efficiency
        phi_mul = np.zeros((self.nlen_f, ntaper))
        abs_mul = np.zeros((self.nlen_f, ntaper))
        dtau_mul = np.zeros((self.nlen_f, ntaper))
        dlna_mul = np.zeros((self.nlen_f, ntaper))
        ephi_ave = np.zeros(self.nlen_f)
        eabs_ave = np.zeros(self.nlen_f)
        edtau_ave = np.zeros(self.nlen_f)
        edlna_ave = np.zeros(self.nlen_f)
        err_phi = np.zeros(self.nlen_f)
        err_abs = np.zeros(self.nlen_f)
        err_dtau = np.zeros(self.nlen_f)
        err_dlna = np.zeros(self.nlen_f)

        # Loop through all tapers
        for itaper in range(0, ntaper):
            # Delete one taper at a time
            tapers_om = np.zeros((nlen_t, ntaper - 1))
            tapers_om[0:self.nlen_f, 0:ntaper - 1] = \
                np.delete(tapers, itaper, 1)

            # FIXME Recalculate MT measurements with deleted taper list
            phi_om, abs_om, dtau_om, dlna_om = self.calculate_multitaper(
                d=d, s=s, tapers=tapers_om, wvec=wvec, nfreq_min=nfreq_min,
                nfreq_max=nfreq_max, cc_tshift=cc_tshift, cc_dlna=cc_dlna
            )

            phi_mul[0:self.nlen_f, itaper] = phi_om[0:self.nlen_f]
            abs_mul[0:self.nlen_f, itaper] = abs_om[0:self.nlen_f]
            dtau_mul[0:self.nlen_f, itaper] = dtau_om[0:self.nlen_f]
            dlna_mul[0:self.nlen_f, itaper] = dlna_om[0:self.nlen_f]

            # Error estimation
            ephi_ave[nfreq_min: nfreq_max] = (
                    ephi_ave[nfreq_min: nfreq_max] +
                    ntaper * phi_mtm[nfreq_min: nfreq_max] -
                    (ntaper - 1) * phi_mul[nfreq_min: nfreq_max, itaper]
            )
            eabs_ave[nfreq_min:nfreq_max] = (
                    eabs_ave[nfreq_min: nfreq_max] +
                    ntaper * abs_mtm[nfreq_min: nfreq_max] -
                    (ntaper - 1) * abs_mul[nfreq_min: nfreq_max, itaper]
            )
            edtau_ave[nfreq_min: nfreq_max] = (
                    edtau_ave[nfreq_min: nfreq_max] +
                    ntaper * dtau_mtm[nfreq_min: nfreq_max] -
                    (ntaper - 1) * dtau_mul[nfreq_min: nfreq_max, itaper]
            )
            edlna_ave[nfreq_min: nfreq_max] = (
                    edlna_ave[nfreq_min: nfreq_max] +
                    ntaper * dlna_mtm[nfreq_min: nfreq_max] -
                    (ntaper - 1) * dlna_mul[nfreq_min: nfreq_max, itaper]
            )

        # Take average over each taper band
        ephi_ave /= ntaper
        eabs_ave /= ntaper
        edtau_ave /= ntaper
        edlna_ave /= ntaper

        # Calculate deviation
        for itaper in range(0, ntaper):
            err_phi[nfreq_min:nfreq_max] += \
                (phi_mul[nfreq_min: nfreq_max, itaper] -
                 ephi_ave[nfreq_min: nfreq_max]) ** 2
            err_abs[nfreq_min:nfreq_max] += \
                (abs_mul[nfreq_min: nfreq_max, itaper] -
                 eabs_ave[nfreq_min: nfreq_max]) ** 2
            err_dtau[nfreq_min:nfreq_max] += \
                (dtau_mul[nfreq_min: nfreq_max, itaper] -
                 edtau_ave[nfreq_min: nfreq_max]) ** 2
            err_dlna[nfreq_min:nfreq_max] += \
                (dlna_mul[nfreq_min: nfreq_max, itaper] -
                 edlna_ave[nfreq_min: nfreq_max]) ** 2

        # Calculate standard deviation
        err_phi[nfreq_min: nfreq_max] = np.sqrt(
            err_phi[nfreq_min:  nfreq_max] / (ntaper * (ntaper - 1)))
        err_abs[nfreq_min: nfreq_max] = np.sqrt(
            err_abs[nfreq_min:  nfreq_max] / (ntaper * (ntaper - 1)))
        err_dtau[nfreq_min: nfreq_max] = np.sqrt(
            err_dtau[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))
        err_dlna[nfreq_min: nfreq_max] = np.sqrt(
            err_dlna[nfreq_min: nfreq_max] / (ntaper * (ntaper - 1)))

        return err_phi, err_abs, err_dtau, err_dlna

    def calculate_freq_limits(self, df):
        """
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
        """
        # Calculate the frequency limits based on FFT of synthetics
        fnum = int(self.nlen_f / 2 + 1)
        s_spectra = np.fft.fft(self.synthetic.data, self.nlen_f) * self.dt

        # Calculate the maximum amplitude of the spectra for the given frequency
        ampmax = max(abs(s_spectra[0: fnum]))
        i_ampmax = np.argmax(abs(s_spectra[0: fnum]))

        # Scale the maximum amplitude by some constant water level
        scaled_wl = self.config.water_threshold * ampmax

        # Default starting values for min/max freq. bands
        ifreq_min = int(1.0 / (self.config.max_period * df))  # default fmin
        ifreq_max = int(1.0 / (self.config.min_period * df))  # default fmax

        # Get the maximum frequency limit by searching valid frequencies
        nfreq_max = fnum - 1
        is_search = True
        for iw in range(0, fnum):
            if iw > i_ampmax:
                nfreq_max, is_search = self._search_frequency_limit(
                        is_search=is_search, index=iw, nfreq_limit=nfreq_max,
                        spectra=s_spectra, water_threshold=scaled_wl
                )
        # Make sure `nfreq_max` does not go beyond the Nyquist frequency
        nfreq_max = min(nfreq_max, ifreq_max, int(1.0 / (2 * self.dt) / df) - 1)

        # Get the minimum frequency limit by searchjing valid frequencies
        nfreq_min = 0
        is_search = True
        for iw in range(fnum - 1, 0, -1):
            if iw < i_ampmax:
                nfreq_min, is_search = self._search_frequency_limit(
                        is_search=is_search, index=iw, nfreq_limit=nfreq_min,
                        spectra=s_spectra, water_threshold=scaled_wl
                )

        # Limit `nfreq_min` by assuming at least N cycles within the window
        nfreq_min = max(
            nfreq_min, ifreq_min,
            int(self.config.min_cycle_in_window / self.tlen_data / df) - 1
        )

        # Reject mtm if the chosen frequency band is narrower than quarter of
        # the multi-taper bandwidth
        half_taper_bandwidth = self.config.mt_nw / (4.0 * self.tlen_data)
        chosen_bandwidth = (nfreq_max - nfreq_min) * df

        if chosen_bandwidth < half_taper_bandwidth:
            logger.debug(f"chosen bandwidth ({chosen_bandwidth}) < "
                         f"half taper bandwidth ({half_taper_bandwidth})")
            nfreq_min = None
            nfreq_max = None
            is_mtm = False
        else:
            is_mtm = True

        return nfreq_min, nfreq_max, is_mtm

    def prepare_data_for_mtm(self, d, tshift, dlna, window):
        """
        Re-window observed data to center on the optimal time shift, and
        scale by amplitude anomaly to get best matching waveforms for MTM

        :return:
        """
        left_sample, right_sample, nlen_w = get_window_info(window, self.dt)
        ishift = int(tshift / self.dt)  # time shift in samples

        left_sample_d = max(left_sample + ishift, 0)
        right_sample_d = min(right_sample + ishift, self.nlen_data)
        nlen_d = right_sample_d - left_sample_d

        if nlen_d == nlen_w:
            # TODO: No need to correct `cc_dlna` in multitaper measurements?
            d[0:nlen_w] = self.observed.data[left_sample_d:right_sample_d]
            d *= np.exp(-dlna)
            window_taper(d, taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)
            is_mtm = True

        # If the shifted time window is now out of bounds of the time series
        # we will not be able to use MTM
        else:
            is_mtm = False

        return d, is_mtm

    def check_time_series_acceptability(self, cc_tshift, nlen_w):
        """
        Checking acceptability of the time series characteristics for MTM

        :type cc_tshift: float
        :param cc_tshift: time shift in unit [s]
        :type nlen_w: int
        :param nlen_w: window length in samples
        :rtype: bool
        :return: True if time series OK for MTM, False if fall back to CC
        """
        # Check length of the time shift w.r.t time step
        if abs(cc_tshift) <= self.dt:
            logger.info(f"reject MTM: time shift {cc_tshift} <= "
                        f"dt ({self.dt})")
            return False

        # Check for sufficient number of wavelengths in window
        elif bool(self.config.min_cycle_in_window * self.config.min_period >
                nlen_w):
            logger.info("reject MTM: too few cycles within time window")
            logger.debug(f"min_period: {self.config.min_period:.2f}s; "
                         f"window length: {nlen_w:.2f}s")
            return False
        else:
            return True

    def check_mtm_time_shift_acceptability(self, nfreq_min, nfreq_max, df,
                                           cc_tshift, dtau_mtm, sigma_dtau_mt):
        """
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
        """
        # True unless set False
        is_mtm = True

        # If any MTM measurements is out of the resonable range, switch to CC
        for j in range(nfreq_min, nfreq_max):
            # dt larger than 1/dt_fac of the wave period
            if np.abs(dtau_mtm[j]) > 1. / (self.config.dt_fac * j * df):
                logger.info("reject MTM: `dt` measurements is too large")
                is_mtm = False

            # Error larger than 1/err_fac of wave period
            if sigma_dtau_mt[j] > 1. / (self.config.err_fac * j * df):
                logger.debug("reject MTM: `dt` error is too large")
                is_mtm = False

            # dt larger than the maximum allowable time shift
            if np.abs(dtau_mtm[j]) > self.config.dt_max_scale * abs(cc_tshift):
                logger.debug("reject MTM: dt is larger than the maximum "
                             "allowable time shift")
                is_mtm = False

        return is_mtm

    def rewindow(self, data, left_sample, right_sample, shift):
        """
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
        """
        nlen_data = len(data)
        nlen = right_sample - left_sample
        lindex = 0

        left_shifted = left_sample + shift
        if left_shifted < 0:
            logger.warn("Re-windowing due to left shift is out of bounds.")
            lindex = -1 * left_shifted
            left_shifted = 0

        rindex = nlen
        right_shifted = right_sample + shift
        if right_shifted > nlen_data:
            logger.warn("Re-windowing due to right shift is out of bounds.")
            rindex = rindex - (right_shifted - nlen_data)
            right_shifted = nlen_data

        data_shifted = np.zeros(nlen)
        data_shifted[lindex:rindex] = data[left_shifted:right_shifted]

        return data_shifted, left_shifted, right_shifted

    @staticmethod
    def _search_frequency_limit(is_search, index, nfreq_limit, spectra,
                                water_threshold, c=10):
        """
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
        """
        if abs(spectra[index]) < water_threshold and is_search:
            is_search = False
            nfreq_limit = index

        if abs(spectra[index]) > c * water_threshold and not is_search:
            is_search = True
            nfreq_limit = index

        return nfreq_limit, is_search


def calculate_adjoint_source(observed, synthetic, config, windows,
                             observed_2=None, synthetic_2=None, windows_2=None):
    """
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
    """
    if config.double_difference:
        for val in [observed_2, synthetic_2, windows_2]:
            assert val is not None, (
                "Double difference measurements require a second set of "
                "waveforms and windows (`observed_2`, `synthetic_2`, "
                "`windows_2`)"
            )

    # Standard Multitaper Misfit approach, single waveform set
    if config.double_difference is False:
        ret_val_p = {}
        ret_val_q = {}

        # Use the MTM class to generate misfit and adjoint sources
        mtm = MultitaperMisfit(observed=observed, synthetic=synthetic,
                               config=config, windows=windows)

        misfit_sum_p, misfit_sum_q, fp, fq, stats = \
                mtm.calculate_adjoint_source()

        # Append information on the misfit for phase and amplitude
        ret_val_p["misfit"] = misfit_sum_p
        ret_val_q["misfit"] = misfit_sum_q

        # Reverse adjoint source in time w.r.t synthetics
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

        if config.measure_type == "dt":
            ret_val = ret_val_p
        elif config.measure_type == "am":
            ret_val = ret_val_q

        ret_val["window_stats"] = stats

    # Double difference multitaper misfit, two sets of waveforms
    elif config.double_difference is True:
        ret_val = {}

        # Use the MTM class to generate misfit and adjoint sources
        mtm = MultitaperMisfit(observed=observed, synthetic=synthetic,
                               config=config, windows=windows, 
                               observed_2=observed_2, synthetic_2=synthetic_2,
                               windows_2=windows_2)
        misfit_sum_p, fp, fp_2, stats = mtm.calculate_dd_adjoint_source()

        ret_val["misfit"] = misfit_sum_p
        ret_val["adjoint_source"] = fp[::-1]
        ret_val["adjoint_source_2"] = fp_2[::-1]
        ret_val["window_stats"] = stats
    else:
        raise NotImplementedError

    return ret_val
