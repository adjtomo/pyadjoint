"""
Main processing scripts to calculate adjoint sources based on two waveforms
"""
import matplotlib.pyplot as plt
import numpy as np
import warnings

from pyadjoint import PyadjointError, PyadjointWarning
from pyadjoint.adjoint_source import AdjointSource
from pyadjoint.utils import sanity_check_waveforms, discover_adjoint_sources


def calculate_adjoint_source(adj_src_type, observed, synthetic, config,
                             window, adjoint_src=True, plot=False,
                             plot_filename=None, **kwargs):
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
    :type window: list of tuples
    :param window: [(left, right),...] representing left and right window
        borders to be tapered in units of seconds since first sample in data
        array
    :param adjoint_src: Derive the adjoint source in addition to misfit calc.
    :type adjoint_src: bool
    :param plot: Also produce a plot of the adjoint source. This will force
        the adjoint source to be calculated regardless of the value of
        ``adjoint_src``.
    :type plot: bool or empty :class:`matplotlib.figure.Figure` instance
    :param plot_filename: If given, the plot of the adjoint source will be
        saved there. Only used if ``plot`` is ``True``.
    :type plot_filename: str
    """
    observed, synthetic = sanity_check_waveforms(observed, synthetic)

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
                      window=window, adjoint_src=adjoint_src, figure=figure,
                      **kwargs)
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

    # Be very defensive and check all the returned parts of the adjoint source.
    # This assures future adjoint source types can be integrated smoothly.
    if adjoint_src:
        adjoint_source = ret_val["adjoint_source"]
        if not isinstance(adjoint_source, np.ndarray) or \
                adjoint_source.dtype != np.float64:
            raise PyadjointError("The adjoint source calculated by the "
                                 "underlying function is no numpy array with "
                                 "a `float64` dtype.")
        if len(adjoint_source.shape) != 1:
            raise PyadjointError(
                "The underlying function returned at adjoint source with "
                "shape %s. It must return a one-dimensional array." % str(
                    adjoint_source.shape))
        if len(adjoint_source) != npts:
            raise PyadjointError(
                "The underlying function returned an adjoint source with %i "
                "samples. It must return a function with %i samples which is "
                "the sample count of the input data." % (
                    len(adjoint_source), npts))
        # Make sure the data returned has no infs or NaNs.
        if not np.isfinite(adjoint_source).all():
            raise PyadjointError(
                "The underlying function returned an adjoint source with "
                "either NaNs or Inf values. This must not be.")
    else:
        adjoint_source = None

    return AdjointSource(adj_src_type, misfit=misfit,
                         adjoint_source=adjoint_source,
                         dt=observed.stats.delta,
                         min_period=config.min_period,
                         max_period=config.max_period,
                         network=observed.stats.network,
                         station=observed.stats.station,
                         component=observed.stats.channel,
                         location=observed.stats.location,
                         starttime=observed.stats.starttime)
