#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration object for pyadjoint.

:copyright:
    Youyi Ruan (youyir@princeton.edu), 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class Config(object):
    def __init__(self, min_period, max_period,
                 lnpt=15,
                 transfunc_waterlevel=1.0E-10,
                 water_threshold=0.02,
                 ipower_costaper=10,
                 min_cycle_in_window=0.5,
                 taper_type='hann',
                 taper_percentage=0.3,
                 mt_nw=4.0,
                 num_taper=5,
                 dt_fac=2.0,
                 phase_step=1.5,
                 err_fac=2.5,
                 dt_max_scale=3.5,
                 measure_type='dt',
                 dt_sigma_min=1.0,
                 dlna_sigma_min=0.5,
                 wtr_env=0.20,
                 use_cc_error=True,
                 use_mt_error=False):
        """
        Central configuration object for Pyadjoint.


        config = Config(
                 min_period, max_period,
                 lnpt = 15,
                 transfunc_waterlevel=1.0E-10,
                 water_threshold=0.02,
                 ipower_costaper=10,
                 min_cycle_in_window=3,
                 taper_type='hann',
                 taper_percentage=0.15,
                 mt_nw=4.0,
                 num_taper=5,
                 phase_step=1.5,
                 dt_fac=2.0,
                 err_fac=2.5,
                 dt_max_scale=3.5,
                 measure_type='dt',
                 dt_sigma_min=1.0,
                 dlna_sigma_min=0.5,
                 wtr_env=0.20,
                 use_cc_error=False,
                 use_mt_error=False)

        :param min_period: Minimum period of the filtered input data in
            seconds.
        :type min_period: float

        :param max_period: Maximum period of the filtered input data in
            seconds.
        :type max_period: float

        :param lnpt: power index to determin the time lenght use in
            FFT (2^lnpt)
        :type lnpt: int

        :param transfunc_waterlevel: Water level on the transfer function
            between data and synthetic.
        :type transfunc_waterlevel: float

        :param ipower_costaper: order of cosine taper, higher the value,
            steeper the shoulders.
        :type ipower_costaper: int

        :param min_cycle_in_window:  Minimum cycle of a wave in time window to
            determin the maximum period can be reliably measured.
        :type min_cycle_in_window: int

        :param taper_percentage: Percentage of a time window needs to be
            tapered at two ends, to remove the non-zero values for adjoint
            source and for fft.
        :type taper_percentage: float

        :param taper_type: Taper type, supports
            "hann", "cos", "cos_p10" so far
        :type taper_type: str

        :param mt_nw: bin width of multitapers (nw*df is the the half
            bandwidth of multitapers in frequency domain,
            typical values are 2.5, 3., 3.5, 4.0)
        :type mt_nw: float

        :param num_taper: number of eigen tapers (2*nw - 3 gives tapers
            with eigen values larger than 0.96)
        :type num_taper: int

        :param dt_fac
        :type dt_fac: float

        :param err_fac
        :type err_fac: float

        :param dt_max_scale
        :type dt_max_scale: float

        :param phase_step: maximum step for cycle skip correction (?)
        :type phase_step: float

        :param dt_sigma_min: minimum travel time error allowed
        :type dt_sigma_min: float

        :param dlna_sigma_min: minimum amplitude error allowed
        :type dlna_sigma_min: float

        :param wtr_env: water level for envelope in denominator
        :type wtr_env: float

        :param measure_type: type of measurements:
                                dt(travel time),
                                am(dlnA),
                                wf(full waveform)
        :param measure_type: string

        :param use_cc_error: use cross correlation errors for
        :type use_cc_error: logic

        :param use_mt_error: use multi-taper error
        :type use_mt_error: logic
        """

        self.min_period = min_period
        self.max_period = max_period

        self.lnpt = lnpt

        self.transfunc_waterlevel = transfunc_waterlevel
        self.water_threshold = water_threshold

        self.ipower_costaper = ipower_costaper

        self.min_cycle_in_window = min_cycle_in_window

        self.taper_type = taper_type
        self.taper_percentage = taper_percentage

        self.mt_nw = mt_nw
        self.num_taper = num_taper
        self.phase_step = phase_step

        self.dt_fac = dt_fac
        self.err_fac = err_fac
        self.dt_max_scale = dt_max_scale

        self.dt_sigma_min = dt_sigma_min
        self.dlna_sigma_min = dlna_sigma_min

        self.wtr_env = wtr_env

        self.measure_type = measure_type
        self.use_cc_error = use_cc_error
        self.use_mt_error = use_mt_error
