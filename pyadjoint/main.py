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
                             windows, plot=False, plot_filename=None,
                             choice=None, observed_2=None,
                             synthetic_2=None, windows_2=None, **kwargs):
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
    :param plot: Also produce a plot of the adjoint source. This will force
        the adjoint source to be calculated regardless of the value of
        ``adjoint_src``.
    :type plot: bool or empty :class:`matplotlib.figure.Figure` instance
    :param plot_filename: If given, the plot of the adjoint source will be
        saved there. Only used if ``plot`` is ``True``.
    :type plot_filename: str
    :type choice: str
    :param choice: Flag to turn on station pair calculations. Requires
        `observed_2`, `synthetic_2`, `windows_2`. Available:
        - 'double_difference': Double difference waveform misfit from
            Yuan et al. 2016
        - 'convolved': Waveform convolution misfit from Choi & Alkhalifah (2011)
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
    observed, synthetic = sanity_check_waveforms(observed, synthetic)

    # Check to see if we're doing double difference
    if choice is not None:
        for check in [observed_2, synthetic_2, windows_2]:
            assert(check is not None), (
                f"setting `choice` requires `observed_2`, `synthetic_2`, "
                f"and `windows_2`")
        observed_2, synthetic_2 = sanity_check_waveforms(observed_2,
                                                         synthetic_2)

    # Require adjoint source to be saved if we're plotting
    if plot:
        assert(plot_filename is not None), f"`plot` requires `plot_filename`"
        adjoint_src = True

    # Get number of samples now as the adjoint source calculation function
    # are allowed to mess with the trace objects.
    npts = observed.stats.npts
    adj_srcs = discover_adjoint_sources()
    if adj_src_type not in adj_srcs.keys():
        raise PyadjointError(f"Adjoint Source type '{adj_src_type}' is unknown."
                             f" Available types: {sorted(adj_srcs.keys())}")

    # From here on out we use this generic function to describe adjoint source
    fct = adj_srcs[adj_src_type][0]

    # Main processing function, calculate adjoint source here
    ret_val = fct(observed=observed, synthetic=synthetic, config=config,
                  windows=windows, choice=choice, observed_2=observed_2,
                  synthetic_2=synthetic_2, windows_2=windows_2, **kwargs)

    # Generate figure from the adjoint source
    if plot:
        figure = plt.figure(figsize=(12, 6))

        # Plot the adjoint source, window and waveforms
        plot_adjoint_source(observed, synthetic, ret_val["adjoint_source"],
                            ret_val["misfit"], windows, adj_src_type)
        figure.savefig(plot_filename)
        plt.show()
        plt.close("all")

        # Plot the double-difference figure if requested
        if choice == "double_difference":
            figure = plt.figure(figsize=(12, 6))
            plot_adjoint_source(observed_2, synthetic_2,
                                ret_val["adjoint_source_2"],
                                ret_val["misfit"], windows_2,
                                f"{adj_src_type}_2")
            fid, ext = os.path.splitext(plot_filename)
            figure.savefig(f"{fid}_2{ext}")
            plt.show()
            plt.close("all")

    # Get misfit and warn for a negative one.
    misfit = float(ret_val["misfit"])
    if misfit < 0.0:
        warnings.warn("Negative misfit value not expected", PyadjointWarning)

    if "adjoint_source" not in ret_val:
        raise PyadjointError("The actual adjoint source was not calculated "
                             "by the underlying function although it was "
                             "requested.")
    if choice == "double_difference":
        try:
            assert("adjoint_source_2" in ret_val)
        except AssertionError:
            raise PyadjointError("The double difference adjoint source was not "
                                 "calculated by the underlying function "
                                 "although it was requested.")

    # Be very defensive and check all the returned parts of the adjoint source.
    # This assures future adjoint source types can be integrated smoothly.
    adjoint_sources = [ret_val["adjoint_source"]]
    # Allow checking double difference adjoint source if present
    if "adjoint_source_2" in ret_val:
        adjoint_sources.append(ret_val["adjoint_source_2"])

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

    adjsrc = AdjointSource(
        adj_src_type, misfit=misfit, dt=observed.stats.delta,
        adjoint_source=ret_val["adjoint_source"], windows=windows,
        min_period=config.min_period, max_period=config.max_period,
        network=observed.stats.network, station=observed.stats.station,
        component=observed.stats.channel, location=observed.stats.location,
        starttime=observed.stats.starttime, window_stats=ret_val["window_stats"]
    )
    if "adjoint_source_2" in ret_val:
        adjsrc_2 = AdjointSource(
            adj_src_type, misfit=misfit, dt=observed.stats.delta,
            adjoint_source=ret_val["adjoint_source_2"], windows=windows_2,
            min_period=config.min_period, max_period=config.max_period,
            network=observed_2.stats.network,
            station=observed_2.stats.station,
            component=observed_2.stats.channel,
            location=observed_2.stats.location,
            starttime=observed.stats.starttime, 
            window_stats=ret_val["window_stats"]
        )
        return adjsrc, adjsrc_2
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
    :type misfit: float
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
