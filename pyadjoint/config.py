#!/usr/bin/env python3
"""
Configuration object for Pyadjoint.

:authors:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Youyi Ruan (youyir@princeton.edu), 2016
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from pyadjoint import discover_adjoint_sources
from pyadjoint.utils.signal import TAPER_COLLECTION


def get_config(adjsrc_type, min_period, max_period, **kwargs):
    """
    Defines two common parameters for all configuration objects and then
    reassigns self to a sub Config class which dictates its own required
    parameters
    """
    adjsrc_type = adjsrc_type.lower()  # allow for case-insensitivity
    adjsrc_types = discover_adjoint_sources().keys()

    assert(min_period < max_period), f"`min_period` must be < `max_period`"

    if adjsrc_type == "waveform_misfit":
        cfg = ConfigWaveform(min_period, max_period, **kwargs)
    elif adjsrc_type == "exponentiated_phase_misfit":
        cfg = ConfigExponentiatedPhase(min_period, max_period, **kwargs)
    elif adjsrc_type == "cc_traveltime_misfit":
        cfg = ConfigCCTraveltime(min_period, max_period, **kwargs)
    elif adjsrc_type == "multitaper_misfit":
        cfg = ConfigMultitaper(min_period, max_period, **kwargs)
    else:
        raise NotImplementedError(f"adjoint source type must be in "
                                  f"{adjsrc_types}")

    # Set the adjoint source type as an attribute for check functions and plots
    cfg.adjsrc_type = adjsrc_type

    # Perform some parameter checks
    if "measure_type" in vars(cfg):
        assert(cfg.measure_type in ["dt", "am"]), \
            "`measure_type` must be 'dt' or 'am'"
    assert(cfg.taper_type in TAPER_COLLECTION), \
        f"`taper_type` must be in {TAPER_COLLECTION}"

    return cfg


class ConfigWaveform:
    """
    Waveform misfit function required parameters

    :param min_period: Minimum period of the filtered input data in seconds.
    :type min_period: float
    :param max_period: Maximum period of the filtered input data in seconds.
    :type max_period: float
    :param taper_percentage: Percentage of a time window needs to be
    tapered at two ends, to remove the non-zero values for adjoint
    source and for fft.
    :type taper_percentage: float
    :param taper_type: Taper type, see `pyaadjoint.utils.TAPER_COLLECTION`
        for a list of available taper types
    :type taper_type: str
    """
    def __init__(self, min_period, max_period, taper_type="hann",
                 taper_percentage=0.3):
        self.min_period = min_period
        self.max_period = max_period
        self.taper_type = taper_type
        self.taper_percentage = taper_percentage


class ConfigExponentiatedPhase:
    """
    Exponentiated Phase misfit function required parameters

    :param min_period: Minimum period of the filtered input data in seconds.
    :type min_period: float
    :param max_period: Maximum period of the filtered input data in seconds.
    :type max_period: float
    :param taper_percentage: Percentage of a time window needs to be
        tapered at two ends, to remove the non-zero values for adjoint
        source and for fft.
    :type taper_percentage: float
    :param taper_type: Taper type, see `pyaadjoint.utils.TAPER_COLLECTION`
        for a list of available taper types
    :type taper_type: str
    :param wtr_env: float
    :param wtr_env: window taper envelope amplitude scaling
    """
    def __init__(self, min_period, max_period, taper_type="hann",
                 taper_percentage=0.3, wtr_env=0.2):
        self.min_period = min_period
        self.max_period = max_period
        self.taper_type = taper_type
        self.taper_percentage = taper_percentage
        self.wtr_env = wtr_env


class ConfigCCTraveltime:
    """
    Cross-correlation Traveltime misfit function required parameters

    :param min_period: Minimum period of the filtered input data in seconds.
    :type min_period: float
    :param max_period: Maximum period of the filtered input data in seconds.
    :type max_period: float
    :param taper_percentage: Percentage of a time window needs to be
    tapered at two ends, to remove the non-zero values for adjoint
    source and for fft.
    :type taper_percentage: float
    :param taper_type: Taper type, see `pyaadjoint.utils.TAPER_COLLECTION`
        for a list of available taper types
    :type taper_type: str
    :param measure_type: measurement type used in calculation of misfit,
        dt(travel time), am(dlnA), wf(full waveform)
    :param measure_type: string
    :param use_cc_error: use cross correlation errors for normalization
    :type use_cc_error: bool
    :param dt_sigma_min: minimum travel time error allowed
    :type dt_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    :type dlna_sigma_min: float
    """
    def __init__(self, min_period, max_period, taper_type="hann",
                 taper_percentage=0.3, measure_type="dt", use_cc_error=True,
                 dt_sigma_min=1.0, dlna_sigma_min=0.5):
        self.min_period = min_period
        self.max_period = max_period
        self.taper_type = taper_type
        self.taper_percentage = taper_percentage
        self.measure_type = measure_type
        self.use_cc_error = use_cc_error
        self.dt_sigma_min = dt_sigma_min
        self.dlna_sigma_min = dlna_sigma_min


class ConfigMultitaper:
    """
    Multitaper misfit function required parameters

    :param min_period: Minimum period of the filtered input data in seconds.
    :type min_period: float
    :param max_period: Maximum period of the filtered input data in seconds.
    :type max_period: float
    :param taper_percentage: Percentage of a time window needs to be
    tapered at two ends, to remove the non-zero values for adjoint
    source and for fft.
    :type taper_percentage: float
    :param taper_type: Taper type, see `pyaadjoint.utils.TAPER_COLLECTION`
        for a list of available taper types
    :type taper_type: str
    :param measure_type: measurement type used in calculation of misfit,
        dt(travel time), am(dlnA), wf(full waveform)
    :type measure_type: str
    :param use_cc_error: use cross correlation errors for normalization
    :type use_cc_error: bool
    :param use_mt_error: use multi-taper error for normalization
    :type use_mt_error: bool
    :param dt_sigma_min: minimum travel time error allowed
    :type dt_sigma_min: float
    :param dlna_sigma_min: minimum amplitude error allowed
    :type dlna_sigma_min: float
    :param lnpt: power index to determine the time length use in FFT
        (2^lnpt)
    :type lnpt: int
    :param transfunc_waterlevel: Water level on the transfer function
        between data and synthetic.
    :type transfunc_waterlevel: float
    :param water_threshold: the triggering value to stop the search. If
        the spectra is larger than 10*water_threshold it will trigger the
        search again, works like the heating thermostat.
    :type water_threshold: float
    :param ipower_costaper: order of cosine taper, higher the value,
        steeper the shoulders.
    :type ipower_costaper: int
    :param min_cycle_in_window:  Minimum cycle of a wave in time window to
        determin the maximum period can be reliably measured.
    :type min_cycle_in_window: int
    :param mt_nw: bin width of multitapers (nw*df is the half
        bandwidth of multitapers in frequency domain,
        typical values are 2.5, 3., 3.5, 4.0)
    :type mt_nw: float
    :param num_taper: number of eigen tapers (2*nw - 3 gives tapers
        with eigen values larger than 0.96)
    :type num_taper: int
    :param dt_fac: percentage of wave period at which measurement range is
        too large and MTM reverts to CCTM misfit
    :type dt_fac: float
    :param err_fac: percentange of error at which error is too large
    :type err_fac: float
    :param dt_max_scale: used to calculate maximum allowable time shift
    :type dt_max_scale: float
    :param phase_step: maximum step for cycle skip correction (?)
    :type phase_step: float
    """
    def __init__(self, min_period, max_period, lnpt=15,
                 transfunc_waterlevel=1.0E-10, water_threshold=0.02,
                 ipower_costaper=10, min_cycle_in_window=0.5, taper_type="hann",
                 taper_percentage=0.3, mt_nw=4.0, num_taper=5, dt_fac=2.0,
                 phase_step=1.5, err_fac=2.5, dt_max_scale=3.5,
                 measure_type="dt", dt_sigma_min=1.0, dlna_sigma_min=0.5,
                 use_cc_error=True, use_mt_error=False):
        self.min_period = min_period
        self.max_period = max_period
        self.taper_type = taper_type
        self.taper_percentage = taper_percentage
        self.measure_type = measure_type
        self.use_cc_error = use_cc_error
        self.dt_sigma_min = dt_sigma_min
        self.dlna_sigma_min = dlna_sigma_min

        self.use_mt_error = use_mt_error
        self.lnpt = lnpt
        self.transfunc_waterlevel = transfunc_waterlevel
        self.water_threshold = water_threshold
        self.ipower_costaper = ipower_costaper
        self.min_cycle_in_window = min_cycle_in_window
        self.mt_nw = mt_nw
        self.num_taper = num_taper
        self.phase_step = phase_step
        self.dt_fac = dt_fac
        self.err_fac = err_fac
        self.dt_max_scale = dt_max_scale

