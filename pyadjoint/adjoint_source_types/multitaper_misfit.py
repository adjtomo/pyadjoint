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
from scipy.integrate import simps

from pyadjoint import logger, plot_adjoint_source
from pyadjoint.utils.dpss import dpss_windows
from pyadjoint.utils.cctm import calculate_cc_shift, calculate_cc_adjsrc
from pyadjoint.utils.signal import (window_taper, get_window_info,
                                    process_cycle_skipping)


VERBOSE_NAME = "Multitaper Misfit"

DESCRIPTION = r"""
The misfit :math:`\chi_P(\mathbf{m})` measures
frequency-dependent phase differences
estimated with multitaper approach.
The misfit :math:`\chi_P(\mathbf{m})`
given Earth model :math:`\mathbf{m}`
and a single receiver is
given by

.. math::

    \chi_P (\mathbf{m}) = \frac{1}{2} \int_0^W  W_P(w) \left|
    \frac{ \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})}
    {\sigma_P(w)} \right|^ 2 dw

:math:`\tau^\mathbf{d}(w)` is the frequency-dependent
phase measurement of the observed data;
:math:`\tau^\mathbf{s}(w, \mathbf{m})` the frequency-dependent
phase measurement of the synthetic data.
The function :math:`W_P(w)` denotes frequency-domain
taper corresponding to the frequency range over which
the measurements are assumed reliable.
:math:`\sigma_P(w)` is associated with the
traveltime uncertainty introduced in making measurements,
which can be estimated with cross-correlation method,
or Jackknife multitaper approach.

The adjoint source for the same receiver is given by

.. math::

    f_P^{\dagger}(t) = \sum_k h_k(t)P_j(t)

in which :math:`h_k(t)` is one (the :math:`k`th) of multi-tapers.

.. math::

    P_j(t) = 2\pi W_p(t) * \Delta \tau(t) * p_j(t) \\
    P_j(w) = 2\pi W_p(w) \Delta \tau(w) * p_j(w)   \\
    p_j(w) = \frac{iw s_j}{\sum_k(iw s_k)(iw s_k)^*} \\
    \Delta \tau(w) = \tau^{\mathbf{d}}(w) - \tau^{\mathbf{s}}(w, \mathbf{m})

"""


class MultitaperMisfit:
    """
    A class to house the machinery of the multitaper misfit calculation. This is
    done with a class rather than a function to reduce the amount of unnecessary
    parameter passing between functions.
    """
    def __init__(self, observed, synthetic, config, windows):
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
        """
        assert (config.__class__.__name__ == "ConfigMultitaper"), \
            "Incorrect configuration class passed to CCTraveltime misfit"

        self.observed = observed
        self.synthetic = synthetic
        self.config = config
        self.windows = windows

        # Calculate some information to be used for MTM measurements
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

            # Calculate cross correlation time shift, amplitude anomaly, errors
            # 'd' and 's' are windowed data and synthetic waves, respectively
            d, s, cc_tshift, cc_dlna, sigma_dt_cc, sigma_dlna_cc = \
                calculate_cc_shift(observed=self.observed,
                                   synthetic=self.synthetic,
                                   window=window, **vars(self.config)
                                   )

            # Perform a series of checks to see if MTM is valid for the data
            # This will only loop once, but allows us to break if a check fail
            while is_mtm is True:
                # Check length of the time shift w.r.t time step
                is_mtm = abs(cc_tshift) <= self.dt
                if is_mtm is False:
                    logger.info(f"reject MTM: time shift {cc_tshift} <= "
                                f"dt ({self.dt})")
                    break

                # Check for sufficient number of wavelengths in window
                is_mtm = bool(
                    self.config.min_cycle_in_window * self.config.min_period <
                    nlen_w
                )
                if is_mtm is False:
                    logger.info("reject MTM: too few cycles within time window")
                    logger.debug(f"min_period: {self.config.min_period:.2f}s; "
                                 f"window length: {self.nlen_w:.2f}s")
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
                                 f"{right_sample * self.t}]")
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
                    sigma_dtau_mt=sigma_dtau_mt
                    )
                if is_mtm is False:
                    break

                # We made it! Use MTM for adjoint source calculation
                logger.info("calculating misfit and adjoint source with MTM")
                fp_t, fq_t, misfit_p, misfit_q = self.calculate_mt_adjsrc(
                    d=d, s=s, tapers=tapers,  nfreq_min=nfreq_min,
                    nfreq_max=nfreq_max, df=df, dtau_mtm=dtau_mtm,
                    dlna_mtm=dlna_mtm, err_dt_cc=sigma_dt_cc,
                    err_dlna_cc=sigma_dlna_cc, err_dtau_mt=sigma_dtau_mt,
                    err_dlna_mt=sigma_dlna_mt
                )
                win_stats.append(
                    {"left": left_sample * self.dt,
                     "right": right_sample * self.dt,
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
                     "measurement_type": self.config.measure_type,
                     "misfit_dt": misfit_p,
                     "misfit_dlna": misfit_q,
                     "sigma_dt": sigma_dt_cc,
                     "sigma_dlna": sigma_dlna_cc,
                     "dt": cc_tshift,
                     "dlna": cc_dlna,
                     }
                )

            # Taper windowed adjoint source
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

    def calculate_mt_adjsrc(self, d, s, tapers, nfreq_min, nfreq_max, df,
                            dtau_mtm, dlna_mtm, err_dt_cc, err_dlna_cc,
                            err_dtau_mt, err_dlna_mt):
        """
        Calculate the adjoint source for a multitaper measurement

        :type d: np.array
        :param d: observed data array
        :type s: np.array
        :param s: synthetic data array
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
        :type err_dt_cc: float
        :param err_dt_cc: cross correlation time shift error
        :type err_dlna_cc: float
        :param err_dlna_cc: cross correlation amplitude anomaly error
        :type err_dtau_mt: np.array
        :param err_dtau_mt: phase-dependent timeshift error
        :type err_dlna_mt: np.array
        :param err_dlna_mt: phase-dependent amplitude error
        """
        nlen_t = len(d)
        ntaper = len(tapers[0])

        # Frequency-domain taper based on adjusted frequency band and
        # error estimation. It's not one of the filtering processes that
        # needed to applied to adjoint source but an frequency domain
        # weighting function for adjoint source and misfit function.
        w_taper = np.zeros(self.nlen_f)

        win_taper_len = nfreq_max - nfreq_min
        win_taper = np.ones(win_taper_len)

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

        wp_w = w_taper / ffac
        wq_w = w_taper / ffac

        # Choose wether to use the CC error or to calculate MT errors
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

        # initialization
        bottom_p = np.zeros(self.nlen_f, dtype=complex)
        bottom_q = np.zeros(self.nlen_f, dtype=complex)

        d2_tw = np.zeros((self.nlen_f, ntaper), dtype=complex)
        d2_tvw = np.zeros((self.nlen_f, ntaper), dtype=complex)

        # Multitaper measurements
        for itaper in range(0, ntaper):
            taper = np.zeros(self.nlen_f)
            taper[0:nlen_t] = tapers[0:nlen_t, itaper]

            # multi-tapered measurements
            d2_t = s * taper[0:nlen_t]
            d2_tv = np.gradient(d2_t, self.dt)

            # apply FFT to tapered measurements
            d2_tw[:, itaper] = np.fft.fft(d2_t, self.nlen_f)[:] * self.dt
            d2_tvw[:, itaper] = np.fft.fft(d2_tv, self.nlen_f)[:] * self.dt

            # calculate bottom of adjoint term pj(w) qj(w)
            bottom_p[:] = (
                    bottom_p[:] +
                    d2_tvw[:, itaper] * d2_tvw[:, itaper].conjugate()
            )
            bottom_q[:] = (
                    bottom_q[:] +
                    d2_tw[:, itaper] * d2_tw[:, itaper].conjugate()
            )

        fp_t = np.zeros(nlen_t)
        fq_t = np.zeros(nlen_t)

        for itaper in range(0, ntaper):
            taper = np.zeros(self.nlen_f)
            taper[0: nlen_t] = tapers[0:nlen_t, itaper]

            # calculate pj(w), qj(w)
            p_w = np.zeros(self.nlen_f, dtype=complex)
            q_w = np.zeros(self.nlen_f, dtype=complex)

            p_w[nfreq_min:nfreq_max] = d2_tvw[nfreq_min:nfreq_max, itaper] / \
                                       (bottom_p[nfreq_min:nfreq_max])
            q_w[nfreq_min:nfreq_max] = -d2_tw[nfreq_min:nfreq_max, itaper] / \
                                       (bottom_q[nfreq_min:nfreq_max])

            # calculate weighted adjoint Pj(w), Qj(w) adding measurement
            # dtau dlna
            p_w *= dtau_mtm * wp_w
            q_w *= dlna_mtm * wq_w

            # inverse FFT to weighted adjoint (take real part)
            p_wt = np.fft.ifft(p_w, self.nlen_f).real * 2. / self.dt
            q_wt = np.fft.ifft(q_w, self.nlen_f).real * 2. / self.dt

            # apply tapering to adjoint source
            fp_t[0:nlen_t] += p_wt[0:nlen_t] * taper[0:nlen_t]
            fq_t[0:nlen_t] += q_wt[0:nlen_t] * taper[0:nlen_t]

        # calculate misfit
        dtau_mtm_weigh_sqr = dtau_mtm ** 2 * wp_w
        dlna_mtm_weigh_sqr = dlna_mtm ** 2 * wq_w

        # Integrate with the composite Simpson's rule.
        misfit_p = 0.5 * 2.0 * simps(y=dtau_mtm_weigh_sqr, dx=df)
        misfit_q = 0.5 * 2.0 * simps(y=dlna_mtm_weigh_sqr, dx=df)

        return fp_t, fq_t, misfit_p, misfit_q

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

            # Recalculate MT measurements with deleted taper list
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
                             adjoint_src=True, window_stats=True):
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
    :type adjoint_src: bool
    :param adjoint_src: flag to calculate adjoint source, if False, will only
        calculate misfit
    :type window_stats: bool
    :param window_stats: flag to return stats for individual misfit windows used
        to generate the adjoint source
    """
    ret_val_p = {}
    ret_val_q = {}

    # Use the MTM class to generate misfit and adjoint sources
    mtm = MultitaperMisfit(observed=observed, synthetic=synthetic,
                           config=config, windows=windows)
    misfit_sum_p, misfit_sum_q, fp, fq, stats = mtm.calculate_adjoint_source()

    # Append information on the misfit for phase and amplitude
    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    # Pin some information about each of the windows provided
    if window_stats:
        ret_val_p["window_stats"] = stats
        ret_val_q["window_stats"] = stats

    # Reverse adjoint source in time w.r.t synthetics
    if adjoint_src:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

    if config.measure_type == "dt":
        ret_val = ret_val_p
    elif config.measure_type == "am":
        ret_val = ret_val_q

    return ret_val
