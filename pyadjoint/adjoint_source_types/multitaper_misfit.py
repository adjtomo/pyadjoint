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
from pyadjoint.utils.mtm import dpss_windows, process_cycle_skipping
from pyadjoint.utils.cctm import xcorr_shift, cc_error
from pyadjoint.utils.signal import window_taper


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
    used to avoid passing parameters in and out of functions which done
    previously
    """
    def __init__(self, observed, synthetic, config, window,
                 adjoint_src=True, window_stats=True, plot=False):
        """
        Initialize Multitaper Misfit adjoint source creator
        """
        assert (config.__class__.__name__ == "ConfigMultitaper"), \
            "Incorrect configuration class passed to CCTraveltime misfit"

        self.observed = observed
        self.synthetic = synthetic
        self.config = config
        self.window = window
        self.adjoint_src = adjoint_src
        self.window_stats = window_stats
        self.plot = plot

        # Calculate some important information to be used for MTM
        self.nlen_f = 2 ** self.config.lnpt
        self.nlen_data = len(synthetic.data)  # length in samples
        self.deltat = synthetic.stats.delta  # sampling rate
        self.tlen_data = self.nlen_data * self.deltat  # length in time [s]

        # Empty variables to be used for the return
        self.ret_val_p = {}
        self.ret_val_q = {}
        self.window_stats = []

    def calculate_adjoint_source(self):
        """
        Main processing function to calculate adjoint source for MTM
        """
        # Arrays for adjoint sources w.r.t time shift (p) and amplitude (q)
        fp = np.zeros(self.nlen_data)
        fq = np.zeros(self.nlen_data)
        misfit_sum_p = 0.0
        misfit_sum_q = 0.0

        # Loop over time windows and calculate misfit for each window range
        for wins in self.window:
            left_window_border = wins[0]
            right_window_border = wins[1]

            # `d` and `s` represent the windowed data and synthetics
            d, s, cc_tshift, cc_dlna, sigma_dt_cc, sigma_dlna_cc, \
                is_mtm, nlen_w= self.calculate_cctm_misfit(left_window_border,
                                                           right_window_border
                                                           )

            # Determine frequency bands for FFT
            freq = np.fft.fftfreq(n=self.nlen_f, d=self.observed.stats.delta)
            df = freq[1] - freq[0]
            wvec = freq * 2 * np.pi  # omega vector: angular frequency
            logger.debug("delta_f (frequency sampling): %f " % df)
            # dw = wvec[1] - wvec[0]  # TODO: check again see if dw is not used.

            # Perform a series of checks to see if MTM is valid for the data
            if is_mtm:
                # CHECK 1:
                # Check window if okay for MTM measurements. If so, find min/max
                # frequency limit for calculations
                nfreq_min, nfreq_max, is_mtm = self.frequency_limit(df)
                if is_mtm:
                    # CHECK 2:
                    # Set the Rayleigh bin parameter (determine taper bandwith
                    # in frequency domain): NW (typical values are 2.5,3,3.5,4).
                    # Generate discrete prolate slepian sequences
                    tapert, eigens = dpss_windows(n=nlen_w,
                                                  half_nbw=self.config.mt_nw,
                                                  k_max=self.config.num_taper,
                                                  low_bias=False)
                    if np.isfinite(eigens).all():
                        tapers = tapert.T * np.sqrt(nlen_w)
                        # CHECK 3:
                        # Check if tapers are properly generated. In rare cases
                        # (e.g., [nw=2.5, nlen=61] or [nw=4.0, nlen=15]) certain
                        # eigenvalues can not be found and associated eigen
                        # taper will be NaN
                        phi_mtm, abs_mtm, dtau_mtm, dlna_mtm = self.mt_measure(
                            d=d, s=s, tapers=tapers, nfreq_min=nfreq_min,
                            nfreq_max=nfreq_max, cc_tshift=cc_tshift,
                            cc_dlna=cc_dlna
                        )

                        # Calculate multi-taper error estimation
                        if self.config.use_mt_error:
                            sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, \
                                sigma_dlna_mt = self.mt_error(
                                    d=d, s=s, tapers=tapers, wvec=wvec,
                                    nfreq_min=nfreq_min, nfreq_max=nfreq_max,
                                    cc_tshift=cc_tshift, cc_dlna=cc_dlna,
                                    phi_mtm=phi_mtm, abs_mtm=abs_mtm,
                                    dtau_mtm=dtau_mtm, dlna_mtm=dlna_mtm
                            )

                        # Check one last time if the multitaper measurement
                        # results fail a selection criteria
                        is_mtm = self.mt_measure_select(
                            nfreq_min=nfreq_min, nfreq_max=nfreq_max, df=df,
                            cc_tshift=cc_tshift, dtau_w=dtau_mtm,
                        )
                    else:
                        logger.warning("Error constructing dpss tapers will "
                                       "fall back to CC")
                        logger.debug(f"eigenvalues: {eigens}")
                        is_mtm = False

            # Determine if MTM or CC will be used to calculate misfit/adjsrc
            if is_mtm:
                fp_t, fq_t, misfit_p, misfit_q = self.mt_adj(
                    d=d, s=s, tapers=tapers,  nfreq_min=nfreq_min,
                    nfreq_max=nfreq_max, df=df, dtau_mtm=dtau_mtm,
                    dlna_mtm=dlna_mtm, err_dt_cc=sigma_dt_cc,
                    err_dlna_cc=sigma_dlna_cc, err_dtau_mt=sigma_dtau_mt,
                    err_dlna_mt=sigma_dlna_mt
                )

                self.windows_stats.append(
                    {"dt": np.mean(dtau_mtm[nfreq_min:nfreq_max]),
                     "dlna": np.mean(dlna_mtm[nfreq_min:nfreq_max]),
                     "misfit_dt": misfit_p,
                     "misfit_dlna": misfit_q,
                     "sigma_dt": sigma_dt_cc,
                     "sigma_dlna": sigma_dlna_cc,
                     }
                )
            else:
                # calculate c.c. adjoint source
                fp_t, fq_t, misfit_p, misfit_q = \
                    cc_adj(s, cc_shift, cc_dlna, deltat, sigma_dt_cc,
                           sigma_dlna_cc)
                measure_wins["type"] = "cc"
                cc_dt = cc_shift * deltat
                measure_wins["dt"] = cc_dt
                measure_wins["misfit_dt"] = 0.5 * (cc_dt / sigma_dt_cc) ** 2

                measure_wins["dlna"] = cc_dlna
                measure_wins["misfit_dlna"] = 0.5 * (
                            cc_dlna / sigma_dlna_cc) ** 2

                measure_wins["sigma_dt"] = sigma_dt_cc
                measure_wins["sigma_dlna"] = sigma_dlna_cc


            # Taper signals following the SAC taper command
            window_taper(fp_t[0:nlen], taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
            window_taper(fq_t[0:nlen], taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)

            # return to original location before windowing
            # initialization
            fp_wind = np.zeros(len(synthetic.data))
            fq_wind = np.zeros(len(synthetic.data))

            fp_wind[left_sample: right_sample] = fp_t[0:nlen]
            fq_wind[left_sample: right_sample] = fq_t[0:nlen]

            fp += fp_wind
            fq += fq_wind

            misfit_sum_p += misfit_p
            misfit_sum_q += misfit_q

            measurement.append(measure_wins)

        ret_val_p["misfit"] = misfit_sum_p
        ret_val_q["misfit"] = misfit_sum_q

        # output all measurement results regardless the type indicated
        ret_val_p["measurement"] = measurement
        ret_val_q["measurement"] = measurement

        if adjoint_src:
            # Reverse in time and reverse the actual values.
            ret_val_p["adjoint_source"] = fp[::-1]
            ret_val_q["adjoint_source"] = fq[::-1]

        if config.measure_type == "dt":
            if figure:
                generic_adjoint_source_plot(observed, synthetic,
                                            ret_val_p["adjoint_source"],
                                            ret_val_p["misfit"],
                                            window, VERBOSE_NAME)

            return ret_val_p

        if config.measure_type == "am":
            if figure:
                generic_adjoint_source_plot(observed, synthetic,
                                            ret_val_q["adjoint_source"],
                                            ret_val_q["misfit"],
                                            window, VERBOSE_NAME)

            return ret_val_q

    def calculate_cctm_misfit(self, left_window_border, right_window_border):
        """
        Calculate cross-correlation traveltime misfit for a given window
        """
        # pre-processing of the observed and synthetic
        left_sample = int(np.floor(left_window_border / self.deltat))
        # Length of the window in samples
        nlen_w = int(np.floor((right_window_border - left_window_border) /
                            self.deltat)) + 1
        right_sample = left_sample + nlen_w

        d = np.zeros(nlen_w)
        s = np.zeros(nlen_w)

        d[0: nlen_w] = self.observed.data[left_sample: right_sample]
        s[0: nlen_w] = self.synthetic.data[left_sample: right_sample]

        # Taper signals following the SAC taper command
        window_taper(d, taper_percentage=self.config.taper_percentage,
                     taper_type=self.config.taper_type)
        window_taper(s, taper_percentage=self.config.taper_percentage,
                     taper_type=self.config.taper_type)

        # cross-correlation: note the c.c. value may dramatically change
        # with/without the tapering in some cases.
        cc_shift = xcorr_shift(d, s)
        cc_tshift = cc_shift * self.deltat
        cc_dlna = 0.5 * np.log(sum(d ** 2) / sum(s ** 2))

        # Uncertainty estimate based on cross-correlations
        if self.config.use_cc_error:
            sigma_dt_cc, sigma_dlna_cc = cc_error(
                d1=d, d2=s, deltat=self.deltat, cc_shift=cc_shift,
                cc_dlna=cc_dlna, dt_sigma_min=self.config.dt_sigma_min,
                dlna_sigma_min=self.config.dlna_sigma_min
            )

            logger.debug("cc_dt  : %f +/- %f" % (cc_tshift, sigma_dt_cc))
            logger.debug("cc_dlna: %f +/- %f" % (cc_dlna, sigma_dlna_cc))
        else:
            sigma_dt_cc = 1.0
            sigma_dlna_cc = 1.0

        # Re-window observed to align observed with synthetic for multitaper
        # measurement:
        left_sample_d = max(left_sample + cc_shift, 0)
        right_sample_d = min(right_sample + cc_shift, self.nlen_data)
        nlen_d = right_sample_d - left_sample_d

        if nlen_d == nlen_w:
            # No need to correct cc_dlna in multitaper measurements
            d[0:nlen_w] = self.observed.data[left_sample_d:right_sample_d]
            d *= np.exp(-cc_dlna)
            window_taper(d, taper_percentage=self.config.taper_percentage,
                         taper_type=self.config.taper_type)
            is_mtm = True
        else:
            # when the shifted window is out of bound
            logger.warning(f"win = [{left_window_border}, "
                           f"{right_window_border}]")
            logger.warning(f"CC shift: {cc_tshift}, out of bounds.")
            is_mtm = False

        return (d, s, cc_tshift, cc_dlna, sigma_dt_cc, sigma_dlna_cc,
                is_mtm, nlen_w)

    def mt_adj(self, d, s, tapers, nfreq_min, nfreq_max, df, dtau_mtm, dlna_mtm,
               err_dt_cc, err_dlna_cc, err_dtau_mt, err_dlna_mt):
        """
        Calculate the adjoint source for a multitaper measurement

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
        logger.debug(f"Frequency bound (idx): [{nfreq_min} {nfreq_max - 1}] "
                     f"(Hz) [{df * (nfreq_min - 1)} {df * nfreq_max}]"
                     )
        logger.debug(f"Frequency domain taper normalization coeff: {ffac}")
        logger.debug(f"Frequency domain sampling length df =  {df}")
        if ffac <= 0.0:
            logger.warning("frequency band too narrow:")
            logger.warning(f"fmin={nfreq_min} fmax={nfreq_max} ffac={ffac}")

        wp_w = w_taper / ffac
        wq_w = w_taper / ffac

        if self.config.use_cc_error:
            wp_w /= err_dt_cc ** 2
            wq_w /= err_dlna_cc ** 2

        # mt error
        if self.config.use_mt_error:
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
            d2_tv = np.gradient(d2_t, self.deltat)

            # apply FFT to tapered measurements
            d2_tw[:, itaper] = np.fft.fft(d2_t, self.nlen_f)[:] * self.deltat
            d2_tvw[:, itaper] = np.fft.fft(d2_tv, self.nlen_f)[:] * self.deltat

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
            p_wt = np.fft.ifft(p_w, self.nlen_f).real * 2. / self.deltat
            q_wt = np.fft.ifft(q_w, self.nlen_f).real * 2. / self.deltat

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

    def mt_measure(self, d, s, tapers, wvec, nfreq_min, nfreq_max, cc_tshift,
                   cc_dlna):
        """
        Measure multitaper misfit

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

            d_tw = np.fft.fft(d_t, self.nlen_f) * self.deltat
            s_tw = np.fft.fft(s_t, self.nlen_f) * self.deltat

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

    def mt_measure_select(self, nfreq_min, nfreq_max, df, cc_tshift, dtau_w):
        """
        Check mtm measurement see if the measurements are good to keep,
        otherwise use CC measurement instead

        :type nfreq_max: int
        :param nfreq_max: maximum in frequency domain
        :type nfreq_min: int
        :param nfreq_min: minimum in frequency domain
        :type cc_tshift: float
        :param cc_tshift: c.c. time shift
        :type dtau_w: float
        :param dtau_w: time dependent travel time measurements from mtm
        :type df: float
        :param df: frequency domain sampling length
        """
        # True unless set False
        is_mtm = True

        # Check if the CC measurements is too small
        if abs(cc_tshift) <= self.deltat:
            logger.debug(f"CC time shift less than time domain sample length "
                         f"{self.deltat}")
            is_mtm = False

        # If any MTM measurements is out of the resonable range, switch to CC
        for j in range(nfreq_min, nfreq_max):
            # dt larger than 1/dt_fac of the wave period
            if np.abs(dtau_w[j]) > 1. / (self.config.dt_fac * j * df):
                logger.info("MTM dt measurements is too large")
                is_mtm = False

            # Error larger than 1/err_fac of wave period
            if self.config.err_dt[j] > 1. / (self.config.err_fac * j * df):
                logger.debug("MTM dt error is too large")
                is_mtm = False

            # dt larger than the maximum time shift allowed
            if np.abs(dtau_w[j]) > self.config.dt_max_scale * abs(cc_tshift):
                logger.debug("MTM dt is larger than the maximum allowable time "
                             "shift")
                is_mtm = False

        return is_mtm

    def mt_error(self, d, s, tapers, wvec, nfreq_min, nfreq_max, cc_tshift,
                 cc_dlna, phi_mtm, abs_mtm, dtau_mtm, dlna_mtm):
        """
        Calculate multitaper error with Jackknife MT estimates

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
        """
        nlen_t = len(d)
        ntaper = len(tapers[0])
        logger.debug("Number of tapers used: %d" % ntaper)

        # Jacknife MT estimates
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
            phi_om, abs_om, dtau_om, dlna_om = self.mt_measure(
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

    def frequency_limit(self, df):
        """
        Determine if a given window is suitable for multitaper measurements.
        If so, finds the maximum frequency range for the measurement using a
        spectrum of tapered synthetic waveforms

        First check if the window is suitable for mtm measurements, then
        find the maximum frequency point for measurement using the spectrum of
        tapered synthetics.

        :type df: float
        :param df: step length of frequency bins for FFT
        :rtype: tuple
        :return (float, float, bool);
            (minimumum frequency, maximum frequency, continue with MTM?)
        """
        ifreq_min = int(1.0 / (self.config.max_period * df))
        ifreq_max = int(1.0 / (self.config.min_period * df))

        # reject MTM if a wave of `min_period` contains a number of cycles less
        # than `min_cycle_in_window` in the selected window. If so switch to
        # CC method. In this case frequency limits are not needed.
        if (self.config.min_cycle_in_window * self.config.min_period) > \
                self.tlen_data:
            logger.debug(f"min_period: {self.config.min_period:6.0f} "
                         f"window length: {self.tlen_data:6.0f}")
            logger.debug("MTM: rejecting for too few cycles within time window")
            return ifreq_min, ifreq_max, False

        # If not rejected, calculate the frequency limits based on FFT of syn
        fnum = int(self.nlen_f / 2 + 1)
        s_spectra = np.fft.fft(self.synthetic.data, self.nlen_f) * self.deltat

        # Calculate the maximum amplitude of the spectra for the given frequency
        ampmax = max(abs(s_spectra[0: fnum]))
        i_ampmax = np.argmax(abs(s_spectra[0: fnum]))

        # Scale the maximum amplitude by some constant water level
        self.config.water_threshold *= ampmax

        # ==================== GET MAXIMUM FREQUENCY LIMIT =====================
        nfreq_max = fnum - 1
        is_search = True
        for iw in range(0, fnum):
            if iw > i_ampmax:
                nfreq_max, is_search = \
                    self.search_frequency_limit(
                        is_search=is_search, index=iw, nfreq_limit=nfreq_max,
                        spectra=s_spectra,
                        water_threshold=self.config.water_threshold
                    )
        # Don't go beyond the Nyquist frequency
        nfreq_max = min(nfreq_max,
                        int(1.0 / (2 * self.deltat) / df) - 1, ifreq_max
                        )

        # ==================== GET MINIMUM FREQUENCY LIMIT =====================
        nfreq_min = 0
        is_search = True
        for iw in range(fnum - 1, 0, -1):
            if iw < i_ampmax:
                nfreq_min, is_search = \
                    self.search_frequency_limit(
                        is_search=is_search, index=iw, nfreq_limit=nfreq_min,
                        spectra=s_spectra,
                        water_threshold=self.config.water_threshold
                                                )

        # Assume there are at least N cycles within the window
        nfreq_min = max(
            nfreq_min,
            int(self.config.min_cycle_in_window / self.tlen_data / df) - 1,
            ifreq_min
        )
        # ======================================================================
        # Reject mtm if the chosen frequency band is narrower than quarter of
        # the multi-taper bandwidth
        half_taper_bandwidth = self.config.mt_nw / (4.0 * self.tlen_data)
        chosen_bandwidth = (nfreq_max - nfreq_min) * df

        if chosen_bandwidth < half_taper_bandwidth:
            logger.debug(f"chosen bandwidth: {chosen_bandwidth} < "
                         f"half taper bandwidth: {half_taper_bandwidth}")
            logger.debug(
                "MTM: rejecting for frequency range narrower than half "
                "taper bandwith")

            return ifreq_min, ifreq_max, False

        return nfreq_min, nfreq_max, True

    @staticmethod
    def search_frequency_limit(is_search, index, nfreq_limit, spectra,
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
        :param water_threshold the triggering value to stop the search
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



