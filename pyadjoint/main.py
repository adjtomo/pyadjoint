"""
Main processing scripts to calculate adjoint sources based on two waveforms
"""

import inspect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import obspy
import warnings

from pyadjoint import PyadjointError, PyadjointWarning, discover_adjoint_sources
from pyadjoint.adjoint_source import AdjointSource
from pyadjoint.utils.signal import sanity_check_waveforms


def calculate_adjoint_source(adj_src_type, observed, synthetic, config,
                             windows, adjoint_src=True, window_stats=True,
                             plot=False, plot_filename=None, 
                             double_difference=False, observed_dd=None,
                             synthetic_dd=None, windows_dd=None, **kwargs):
    """
    Central function of Pyadjoint used to calculate adjoint sources and misfit.

    This function uses the notion of observed and synthetic data to offer a
    nomenclature most users are familiar with. Please note that it is
    nonetheless independent of what the two data arrays actually represent.

    The function tapers the data from ``left_window_border`` to
    ``right_window_border``, both in seconds since the first sample in the
    data arrays.

    :param adj_src_type: The type of adjoint source to calculate.
    :type adj_src_type: str
    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param config: :class:`pyadjoint.config.Config`
    :type config: configuration parameters that control measurement
    :type windows: list of tuples
    :param windows: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array
    :param adjoint_src: Derive the adjoint source in addition to misfit calc.
    :type adjoint_src: bool
    :param window_stats: Return stats (misfit, measurement type) for each window
        provided to the adjoint source calculation.
    :param plot: Also produce a plot of the adjoint source. This will force
        the adjoint source to be calculated regardless of the value of
        ``adjoint_src``.
    :type plot: bool or empty :class:`matplotlib.figure.Figure` instance
    :param plot_filename: If given, the plot of the adjoint source will be
        saved there. Only used if ``plot`` is ``True``.
    :type plot_filename: str
        :type double_difference: bool
    :param double_difference: flag to turn on double difference waveform
        misfit measurement. Requires `observed_dd`, `synthetic_dd`, `windows_dd`
    :type observed_dd: obspy.core.trace.Trace
    :param observed_dd: second observed waveform to calculate double difference
        adjoint source
    :type synthetic_dd:  obspy.core.trace.Trace
    :param synthetic_dd: second synthetic waveform to calculate double
        difference adjoint source
    :type windows_dd: list of tuples
    :param windows_dd: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array. Used to window `observed_dd` and `synthetic_dd`
    """
    observed, synthetic = sanity_check_waveforms(observed, synthetic)

    # Check to see if we're doing double difference
    if double_difference:
        for check in [observed_dd, synthetic_dd, windows_dd]:
            assert(check is not None), (
                f"`double_difference` requires `observed_dd`, `synthetic_dd`, "
                f"and `windows_dd`")
        observed_dd, synthetic_dd = sanity_check_waveforms(observed_dd, 
                                                           synthetic_dd)

    # Get number of samples now as the adjoint source calculation function
    # are allowed to mess with the trace objects.
    npts = observed.stats.npts
    adj_srcs = discover_adjoint_sources()
    if adj_src_type not in adj_srcs.keys():
        raise PyadjointError(f"Adjoint Source type '{adj_src_type}' is unknown."
                             f" Available types: {sorted(adj_srcs.keys())}")

    # From here on out we use this generic function to describe adjoint source
    fct = adj_srcs[adj_src_type][0]

    # Set up plotting generation
    if plot:
        # The plot kwargs overwrites the adjoint_src kwarg.
        adjoint_src = True
        if plot is True:
            figure = plt.figure(figsize=(12, 6))
        else:
            # Assume plot is a preexisting figure instance
            figure = plot
    else:
        figure = None

    # Main processing function, calculate adjoint source here
    try:
        ret_val = fct(observed=observed, synthetic=synthetic, config=config,
                      windows=windows, adjoint_src=adjoint_src,
                      window_stats=window_stats, plot=figure, **kwargs)
        # Generate figure from the adjoint source
        if plot and plot_filename:
            figure.savefig(plot_filename)
        elif plot is True:
            plt.show()
    finally:
        # Assure the figure is closed. Otherwise matplotlib will leak memory
        if plot is True:
            plt.close()

    # Get misfit and warn for a negative one.
    misfit = float(ret_val["misfit"])
    if misfit < 0.0:
        warnings.warn("Negative misfit value not expected", PyadjointWarning)

    if adjoint_src and "adjoint_source" not in ret_val:
        raise PyadjointError("The actual adjoint source was not calculated "
                             "by the underlying function although it was "
                             "requested.")
    if adjoint_src and double_difference and "adjoint_source_dd" not in ret_val:
        raise PyadjointError("The double difference adjoint source was not "
                             "calculated by the underlying function "
                             "although it was requested.")

    # Be very defensive and check all the returned parts of the adjoint source.
    # This assures future adjoint source types can be integrated smoothly.
    if adjoint_src:
        adjoint_sources = [ret_val["adjoint_source"]]
        # Allow checking double difference adjoint source if present
        if double_difference:
            adjoint_sources.append(ret_val["adjoint_source_dd"])
        else:
            adjoint_sources.append(None)

        for adjoint_source in adjoint_sources:
            if not isinstance(adjoint_source, np.ndarray) or \
                    adjoint_source.dtype != np.float64:
                raise PyadjointError("The adjoint source calculated by the "
                                     "underlying function is no numpy array "
                                     "with a `float64` dtype.")
            if len(adjoint_source.shape) != 1:
                raise PyadjointError(
                    "The underlying function returned at adjoint source with "
                    f"shape {adjoint_source.shape}. It must return a "
                    "one-dimensional array.")
            if len(adjoint_source) != npts:
                raise PyadjointError(
                    f"The underlying function returned an adjoint source with "
                    f"{len(adjoint_source)} samples. It must return a function "
                    f"with {npts} samples which is the sample count of the "
                    f"input data.")
            # Make sure the data returned has no infs or NaNs.
            if not np.isfinite(adjoint_source).all():
                raise PyadjointError(
                    "The underlying function returned an adjoint source with "
                    "either NaNs or Inf values. This must not be.")
        adjoint_source, adjoint_source_dd = adjoint_sources
    else:
        adjoint_source = None
        adjoint_source_dd = None

    if window_stats:
        windows = ret_val["window_stats"]
    else:
        windows = None

    adjsrc = AdjointSource(
        adj_src_type, misfit=misfit, dt=observed.stats.delta,
        adjoint_source=adjoint_source, windows=windows,
        min_period=config.min_period, max_period=config.max_period,
        network=observed.stats.network, station=observed.stats.station,
        component=observed.stats.channel, location=observed.stats.location,
        starttime=observed.stats.starttime
    )
    if double_difference:
        adjsrc_dd = AdjointSource(
            adj_src_type, misfit=misfit, dt=observed.stats.delta,
            adjoint_source=adjoint_source_dd, windows=windows_dd,
            min_period=config.min_period, max_period=config.max_period,
            network=observed_dd.stats.network,
            station=observed_dd.stats.station,
            component=observed_dd.stats.channel,
            location=observed_dd.stats.location,
            starttime=observed_dd.stats.starttime
        )
        return adjsrc, adjsrc_dd
    else:
        return adjsrc


def plot_adjoint_source(observed, synthetic, adjoint_source, misfit, windows,
                        adjoint_source_name):
    """
    Generic plotting function for adjoint sources and data.

    Many types of adjoint sources can be represented in the same manner.
    This is a convenience function that can be called by different
    the implementations for different adjoint sources.

    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param adjoint_source: The adjoint source.
    :type adjoint_source: `numpy.ndarray`
    :param misfit: The associated misfit value.
    :float misfit: misfit value
    :type windows: list of tuples
    :param windows: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array
    :param adjoint_source_name: The name of the adjoint source.
    :type adjoint_source_name: str
    """
    x_range = observed.stats.endtime - observed.stats.starttime
    left_window_border = 60000.
    right_window_border = 0.

    for window in windows:
        left_window_border = min(left_window_border, window[0])
        right_window_border = max(right_window_border, window[1])

    buf = (right_window_border - left_window_border) * 0.3
    left_window_border -= buf
    right_window_border += buf
    left_window_border = max(0, left_window_border)
    right_window_border = min(x_range, right_window_border)

    plt.subplot(211)
    plt.plot(observed.times(), observed.data, color="0.2", label="Observed",
             lw=2)
    plt.plot(synthetic.times(), synthetic.data, color="#bb474f",
             label="Synthetic", lw=2)
    for window in windows:
        re = patches.Rectangle((window[0], plt.ylim()[0]),
                               window[1] - window[0],
                               plt.ylim()[1] - plt.ylim()[0],
                               color="blue", alpha=0.1)
        plt.gca().add_patch(re)

    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)
    plt.xlim(left_window_border, right_window_border)

    # Determine min and max amplitudes within the time window
    obs_win = observed.data[int(left_window_border):int(right_window_border)]
    syn_win = synthetic.data[int(left_window_border):int(right_window_border)]
    ylim = max([max(np.abs(obs_win)), max(np.abs(syn_win))])
    plt.ylim(-ylim, ylim)

    plt.subplot(212)
    plt.plot(observed.times(), adjoint_source[::-1], color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    # No time reversal for comparison with data
    # plt.xlim(x_range - right_window_border, x_range - left_window_border)
    # plt.xlabel("Time in seconds since first sample")
    plt.xlim(left_window_border, right_window_border)
    plt.xlabel("Time (seconds)")
    ylim = max(map(abs, plt.ylim()))
    plt.ylim(-ylim, ylim)

    plt.suptitle("%s Adjoint Source with a Misfit of %.3g" % (
        adjoint_source_name, misfit))


def get_example_data():
    """
    Helper function returning example data for Pyadjoint.

    The returned data is fully preprocessed and ready to be used with Pyadjoint.

    :returns: Tuple of observed and synthetic streams
    :rtype: tuple of :class:`obspy.core.stream.Stream` objects

    .. rubric:: Example

    >>> from pyadjoint import get_example_data
    >>> observed, synthetic = get_example_data()
    >>> print(observed)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    SY.DBO.S3.MXR | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO.S3.MXT | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO.S3.MXZ | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    >>> print(synthetic)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    3 Trace(s) in Stream:
    SY.DBO..LXR   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO..LXT   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    SY.DBO..LXZ   | 2014-11-15T02:31:50.259999Z - ... | 1.0 Hz, 3600 samples
    """
    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "example_data")
    observed = obspy.read(os.path.join(path, "observed_processed.mseed"))
    observed.sort()
    synthetic = obspy.read(os.path.join(path, "synthetic_processed.mseed"))
    synthetic.sort()

    return observed, synthetic
