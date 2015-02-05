#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Adjoint source class definition.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect
import matplotlib.pylab as plt
import numpy as np
import obspy
import os
import pkgutil
import warnings

from . import PyadjointError, PyadjointWarning


class AdjointSource(object):
    # Dictionary of available adjoint source. The key is the name, the value
    # a tuple of function, verbose name, and description.
    _ad_srcs = {}

    def __init__(self, adj_src_type, misfit, dt, component,
                 adjoint_source=None, network=None, station=None):
        if adj_src_type not in self._ad_srcs:
            raise ValueError("Unknown adjoint source type '%s'." %
                             adj_src_type)
        self.adj_src_type = adj_src_type
        self.adj_src_name = self._ad_srcs[adj_src_type][1]
        self.misfit = misfit
        self.dt = dt
        self.component = component
        self.network = network
        self.station = station
        self.adjoint_source = adjoint_source

    def __str__(self):
        if self.network and self.station:
            station = " at station %s.%s" % (self.network, self.station)
        else:
            station = ""

        if self.adjoint_source is not None:
            adj_src_status = "available with %i samples" % (len(
                self.adjoint_source))
        else:
            adj_src_status = "has not been calculated"

        return (
            "{name} Adjoint Source for component {component}{station}\n"
            "    Misfit: {misfit}\n"
            "    Adjoint source {adj_src_status}"
        ).format(
            name=self.adj_src_name,
            component=self.component,
            station=station,
            misfit=self.misfit,
            adj_src_status=adj_src_status
        )


def calculate_adjoint_source(adj_src_type, observed, synthetic, min_period,
                             max_period, left_window_border,
                             right_window_border, adjoint_src=True,
                             plot=False, plot_filename=None, **kwargs):
    """
    Central function of Pyadjoint used to calculate adjoint sources and misfit.

    This function uses the notion of observed and synthetic data to offer a
    nomenclature most users are familiar with. Please note that it is
    nonetheless independent of what the two data arrays actually represent.

    The function tapers the data from ``left_window_border`` to
    ``right_window_border``, both in seconds since the first sample in the
    data arrays.

    :param adj_src_type: The type of adjoint source to calculate.
    :str adj_src_type: str
    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`
    :param min_period: The minimum period of the spectral content of the data.
    :type min_period: float
    :param max_period: The maximum period of the spectral content of the data.
    :type max_period: float
    :param left_window_border: Left border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type left_window_border: float
    :param left_window_border: Right border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type right_window_border: float
    :param adjoint_src: Only calculate the misfit or also derive
        the adjoint source.
    :type adjoint_src: bool
    :param plot: Also produce a plot of the adjoint source. This will force
        the adjoint source to be calculated regardless of the value of
        ``adjoint_src``.
    :type plot: bool or empty :class:`matplotlib.figure.Figure` instance
    :param plot_filename: If given, the plot of the adjoint source will be
        saved there. Only used if ``plot`` is ``True``.
    :type plot_filename: str
    """
    observed, synthetic = _sanity_checks(observed, synthetic)

    if adj_src_type not in AdjointSource._ad_srcs:
        raise PyadjointError(
            "Adjoint Source type '%s' is unknown. Available types: %s" % (
                adj_src_type, ", ".join(
                    sorted(AdjointSource._ad_srcs.keys()))))

    fct = AdjointSource._ad_srcs[adj_src_type][0]

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
    try:
        ret_val = fct(observed=observed, synthetic=synthetic,
                      min_period=min_period, max_period=max_period,
                      left_window_border=left_window_border,
                      right_window_border=right_window_border,
                      adjoint_src=adjoint_src, figure=figure, **kwargs)

        if plot and plot_filename:
            figure.savefig(plot_filename)
    finally:
        # Assure the figure is closed. Otherwise matplotlib will leak
        # memory. If the figure has been created outside of Pyadjoint,
        # it will not be closed.
        if plot is True:
            plt.close()

    if adjoint_src and "adjoint_source" not in ret_val:
        raise PyadjointError("The actual adjoint source was not calculated "
                             "by the underlying function although it was "
                             "requested.")

    misfit = ret_val["misfit"]
    adjoint_source = ret_val["adjoint_source"] if adjoint_src else None

    return AdjointSource(adj_src_type, misfit=misfit,
                         adjoint_source=adjoint_source,
                         dt=observed.stats.delta,
                         network=observed.stats.network,
                         station=observed.stats.station,
                         component=observed.stats.channel[-1])


def _sanity_checks(observed, synthetic):
    """
    Perform a number of basic sanity checks to assure the data is valid
    in a certain sense.

    It checks the types of both, the start time, sampling rate, number of
    samples, ...

    :param observed: The observed data.
    :type observed: :class:`obspy.core.trace.Trace`
    :param synthetic: The synthetic data.
    :type synthetic: :class:`obspy.core.trace.Trace`

    :raises: :class:`~pyadjoint.PyadjointError`
    """
    if not isinstance(observed, obspy.Trace):
        # Also accept Stream objects.
        if isinstance(observed, obspy.Stream) and \
                len(observed) == 1:
            observed = observed[0]
        else:
            raise PyadjointError(
                "Observed data must be an ObsPy Trace object.")
    if not isinstance(synthetic, obspy.Trace):
        if isinstance(synthetic, obspy.Stream) and \
                len(synthetic) == 1:
            synthetic = synthetic[0]
        else:
            raise PyadjointError(
                "Synthetic data must be an ObsPy Trace object.")

    if observed.stats.npts != synthetic.stats.npts:
        raise PyadjointError("Observed and synthetic data must have the same "
                             "number of samples.")

    sr1 = observed.stats.sampling_rate
    sr2 = synthetic.stats.sampling_rate

    if abs(sr1 - sr2) / sr1 >= 1E-5:
        raise PyadjointError("Observed and synthetic data must have the same "
                             "sampling rate.")

    # Make sure data and synthetics start within half a sample interval.
    if abs(observed.stats.starttime - synthetic.stats.starttime) > \
            observed.stats.delta * 0.5:
        raise PyadjointError("Observed and synthetic data must have the same "
                             "starttime.")

    ptp = sorted([observed.data.ptp(), synthetic.data.ptp()])
    if ptp[1] / ptp[0] >= 5:
        warnings.warn("The amplitude difference between data and "
                      "synthetic is fairly large.", PyadjointWarning)

    # Also check the components of the data to avoid silly mistakes of
    # users.
    if len(set([observed.stats.channel[-1].upper(),
                synthetic.stats.channel[-1].upper()])) != 1:
        warnings.warn("The orientation code of synthetic and observed "
                      "data is not equal.")

    observed = observed.copy()
    synthetic = synthetic.copy()
    observed.data = np.require(observed.data, dtype=np.float64,
                               requirements=["C"])
    synthetic.data = np.require(synthetic.data, dtype=np.float64,
                                requirements=["C"])

    return observed, synthetic


def _discover_adjoint_sources():
    """
    Discovers the available adjoint sources. This should work no matter if
    pyadjoint is checked out from git, packaged as .egg or for any other
    possibility.
    """
    from . import adjoint_source_types

    AdjointSource._ad_srcs = {}

    FCT_NAME = "calculate_adjoint_source"
    NAME_ATTR = "VERBOSE_NAME"
    DESC_ATTR = "DESCRIPTION"

    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        if not hasattr(m, FCT_NAME):
            continue
        fct = getattr(m, FCT_NAME)
        if not callable(fct):
            continue

        name = modname.split('.')[-1]

        if not hasattr(m, NAME_ATTR):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, NAME_ATTR))

        if not hasattr(m, DESC_ATTR):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, DESC_ATTR))

        # Add tuple of name, verbose name, and description.
        AdjointSource._ad_srcs[name] = (
            fct,
            getattr(m, NAME_ATTR),
            getattr(m, DESC_ATTR))


_discover_adjoint_sources()