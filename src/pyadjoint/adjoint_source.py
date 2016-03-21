#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Central interfaces for ``Pyadjoint``.

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
    _ad_srcs_DD = {}

    def __init__(self, adj_src_type, misfit, dt, min_period, max_period,
                 component, adjoint_source=None, network=None, station=None,
                 location=None, starttime=None):
        """
        Class representing an already calculated adjoint source.

        :param adj_src_type: The type of adjoint source.
        :type adj_src_type:  str
        :param misfit: The misfit value.
        :type misfit: float
        :param dt: The sampling rate of the adjoint source.
        :type dt: float
        :param min_period: The minimum period of the spectral content
            of the data.
        :type min_period: float
        :param max_period: The maximum period of the spectral content
            of the data.
        :type max_period: float
        :param component: The adjoint source component, usually ``"Z"``,
            ``"N"``, ``"E"``, ``"R"``, or ``"T"``.
        :type component: str
        :param adjoint_source: The actual adjoint source.
        :type adjoint_source: :class:`numpy.ndarray`
        :param network: The network code of the station.
        :type network: str
        :param station: The station code of the station.
        :type station: str
        :param location: The location code of the station.
        :type location: str
        :param starttime: starttime of adjoint source
        :type starttime: obspy.UTCDateTime
        """

        if adj_src_type in self._ad_srcs:
            self.adj_src_name = self._ad_srcs[adj_src_type][1]
        elif adj_src_type in self._ad_srcs_DD:
            self.adj_src_name = self._ad_srcs_DD[adj_src_type][1]
        else:
            raise ValueError("Unknown adjoint source type '%s'." %
                             adj_src_type)
        self.adj_src_type = adj_src_type
        self.misfit = misfit
        self.dt = dt
        self.min_period = min_period
        self.max_period = max_period
        self.component = component
        self.network = network
        self.station = station
        self.location = location
        self.starttime = starttime
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
            "    Misfit: {misfit:.4g}\n"
            "    Adjoint source {adj_src_status}"
        ).format(
            name=self.adj_src_name,
            component=self.component,
            station=station,
            misfit=self.misfit,
            adj_src_status=adj_src_status
        )

    def write(self, filename, format, **kwargs):
        """
        Write the adjoint source to a file.

        :param filename: Determines where the adjoint source is saved.
        :type filename: str, open file, or file-like object
        :param format: The format of the adjoint source. Currently available
            are: ``"SPECFEM"``
        :type format: str

        .. rubric:: SPECFEM

        SPECFEM requires one additional parameter: the temporal offset of the
        first sample in seconds. The following example sets the time of the
        first sample in the adjoint source to ``-10``.

        >>> adj_src.write("NET.STA.CHAN.adj", format="SPECFEM",
        ...               time_offset=-10)  # doctest: +SKIP
        """
        if self.adjoint_source is None:
            raise ValueError("Can only write adjoint sources if the adjoint "
                             "source has been calculated.")

        format = format.upper()
        available_formats = ["SPECFEM"]
        if format not in available_formats:
            raise ValueError("format '%s' not known. Available formats: %s" %
                             (format, ", ".join(available_formats)))

        if not hasattr(filename, "write"):
            with open(filename, "wb") as fh:
                self._write(fh, format=format, **kwargs)
        else:
            self._write(filename, format=format, **kwargs)

    def _write(self, buf, format, **kwargs):
        if format == "SPECFEM":
            self._write_specfem(buf=buf, time_offset=kwargs["time_offset"])
        else:
            raise NotImplementedError

    def _write_specfem(self, buf, time_offset):
        """
        Write the adjoint source for SPECFEM.
        """
        l = len(self.adjoint_source)

        to_write = np.empty((l, 2))

        to_write[:, 0] = np.linspace(0, (l - 1) * self.dt, l)
        to_write[:, 0] += time_offset
        # SPECFEM expects non-time reversed adjoint sources.
        to_write[:, 1] = self.adjoint_source[::-1]

        np.savetxt(buf, to_write)

    def write_to_asdf(self, ds, time_offset, coordinates=None, **kwargs):
        """
        Writes the adjoint source to an ASDF file.
        Note: For now it is assumed SPECFEM will be using the adjoint source

        :param ds: The ASDF data structure read in using pyasdf.
        :type ds: str
        :param time_offset: The temporal offset of the first sample in seconds.
            This is required if using the adjoint source as input to SPECFEM.
        :type time_offset: float
        :param coordinates: If given, the coordinates of the adjoint source.
            The 'latitude', 'longitude', and 'elevation_in_m' of the adjoint
            source must be defined.
        :type coordinates: list

        .. rubric:: SPECFEM

        SPECFEM requires one additional parameter: the temporal offset of the
        first sample in seconds. The following example sets the time of the
        first sample in the adjoint source to ``-10``.

        >>> adj_src.write_to_asdf(ds, time_offset=-10,
        ...               coordinates={'latitude':19.2,
        ...                            'longitude':13.4,
        ...                            'elevation_in_m':2.0})
        """
        # Import here to not have a global dependency on pyasdf
        from pyasdf.exceptions import NoStationXMLForStation

        # Convert the adjoint source to SPECFEM format
        l = len(self.adjoint_source)
        specfem_adj_source = np.empty((l, 2))
        specfem_adj_source[:, 0] = np.linspace(0, (l - 1) * self.dt, l)
        specfem_adj_source[:, 1] = time_offset
        specfem_adj_source[:, 1] = self.adjoint_source[::-1]

        tag = "%s_%s_%s" % (self.network, self.station, self.component)
        min_period = self.min_period
        max_period = self.max_period
        component = self.component
        station_id = "%s.%s" % (self.network, self.station)

        if coordinates:
            # If given, all three coordinates must be present
            if {"latitude", "longitude", "elevation_in_m"}.difference(
                    set(coordinates.keys())):
                raise ValueError(
                    "'latitude', 'longitude', and 'elevation_in_m'"
                    " must be given")
        else:
            try:
                coordinates = ds.waveforms[
                    "%s.%s" % (self.network, self.station)].coordinates
            except NoStationXMLForStation:
                raise ValueError("Coordinates must either be given "
                                 "directly or already be part of the "
                                 "ASDF file")

        # Safeguard against funny types in the coordinates dictionary
        latitude = float(coordinates["latitude"])
        longitude = float(coordinates["longitude"])
        elevation_in_m = float(coordinates["elevation_in_m"])

        parameters = {"dt": self.dt, "misfit_value": self.misfit,
                      "adjoint_source_type": self.adj_src_type,
                      "min_period": min_period, "max_period": max_period,
                      "latitude": latitude, "longitude": longitude,
                      "elevation_in_m": elevation_in_m,
                      "station_id": station_id, "component": component,
                      "units": "m"}

        # Use pyasdf to add auxiliary data to the ASDF file
        ds.add_auxiliary_data(data=specfem_adj_source,
                              data_type="AdjointSource", path=tag,
                              parameters=parameters)


def calculate_adjoint_source(adj_src_type, observed, synthetic, config,
                             window, adjoint_src=True,
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
    :type adj_src_type: str
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
    :param right_window_border: Right border of the window to be tapered in
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

    # Get number of samples now as the adjoint source calculation function
    # are allowed to mess with the trace objects.
    npts = observed.stats.npts

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
                      config=config, window=window,
                      adjoint_src=adjoint_src, figure=figure, **kwargs)

        if plot and plot_filename:
            figure.savefig(plot_filename)
        elif plot is True:
            plt.show()

    finally:
        # Assure the figure is closed. Otherwise matplotlib will leak
        # memory. If the figure has been created outside of Pyadjoint,
        # it will not be closed.
        if plot is True:
            plt.close()

    # Get misfit an warn for a negative one.
    misfit = float(ret_val["misfit"])
    if misfit < 0.0:
        warnings.warn("The misfit value is negative. Be cautious!",
                      PyadjointWarning)

    if adjoint_src and "adjoint_source" not in ret_val:
        raise PyadjointError("The actual adjoint source was not calculated "
                             "by the underlying function although it was "
                             "requested.")

    # Be very defensive. This assures future adjoint source types can be
    # integrated smoothly.
    if adjoint_src:
        adjoint_source = ret_val["adjoint_source"]
        # Raise if wrong type.
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


def calculate_adjoint_source_DD(adj_src_type, observed1, synthetic1,
                                observed2, synthetic2, config,
                                window1, window2,
                                adjoint_src=True, plot=False,
                                plot_filename=None, **kwargs):

    """
    Central function of Pyadjoint used to calculate double-difference
    adjoint sources and misfit.

    This function uses the notion of observed and synthetic data to offer a
    nomenclature most users are familiar with. Please note that it is
    nonetheless independent of what the two data arrays actually represent.

    The function tapers the data from ``left_window_border`` to
    ``right_window_border``, both in seconds since the first sample in the
    data arrays.

    :param adj_src_type: The type of adjoint source to calculate.
    :type adj_src_type: str
    :param observed1: The observed1 data.
    :type observed1: :class:`obspy.core.trace.Trace`
    :param synthetic1: The synthetic1 data.
    :type synthetic1: :class:`obspy.core.trace.Trace`
    :param observed2: The observed2 data.
    :type observed2: :class:`obspy.core.trace.Trace`
    :param synthetic2: The synthetic2 data.
    :type synthetic2: :class:`obspy.core.trace.Trace`
    :param min_period: The minimum period of the spectral content of the data.
    :type min_period: float
    :param max_period: The maximum period of the spectral content of the data.
    :type max_period: float
    :param left_window_border: Left border of the window to be tapered in
        seconds since the first sample in the data arrays.
    :type left_window_border: float
    :param right_window_border: Right border of the window to be tapered in
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

    observed1, synthetic1 = _sanity_checks(observed1, synthetic1)
    observed2, synthetic2 = _sanity_checks(observed2, synthetic2)

    # Get number of samples now as the adjoint source calculation function
    # are allowed to mess with the trace objects.
    npts = observed1.stats.npts
    if adj_src_type not in AdjointSource._ad_srcs_DD:
        raise PyadjointError(
            "Adjoint Source type '%s' is unknown. Available types: %s" % (
                adj_src_type, ", ".join(
                    sorted(AdjointSource._ad_srcs_DD.keys()))))

    fct = AdjointSource._ad_srcs_DD[adj_src_type][0]
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
        ret_val = fct(observed1=observed1, synthetic1=synthetic1,
                      observed2=observed2, synthetic2=synthetic2,
                      config=config, window1=window1, window2=window2,
                      adjoint_src=adjoint_src, figure=figure, **kwargs)

        if plot and plot_filename:
            figure.savefig(plot_filename)
        elif plot is True:
            plt.show()

    finally:
        # Assure the figure is closed. Otherwise matplotlib will leak
        # memory. If the figure has been created outside of Pyadjoint,
        # it will not be closed.
        if plot is True:
            plt.close()

    # Get misfit an warn for a negative one.
    misfit = float(ret_val["misfit"])
    if misfit < 0.0:
        warnings.warn("The misfit value is negative. Be cautious!",
                      PyadjointWarning)

    if adjoint_src and "adjoint_source" not in ret_val:
        raise PyadjointError("The actual adjoint source was not calculated "
                             "by the underlying function although it was "
                             "requested.")

    # Be very defensive. This assures future adjoint source types can be
    # integrated smoothly.
    if adjoint_src:
        adjoint_source = ret_val["adjoint_source"]
        # Raise if wrong type.
        if not isinstance(adjoint_source, np.ndarray) or \
                adjoint_source.dtype != np.float64:
            raise PyadjointError("The adjoint source 1 calculated by the "
                                 "underlying function is no numpy array with "
                                 "a `float64` dtype.")
        if len(adjoint_source.shape) != 1:
            raise PyadjointError(
                "The underlying function returned at adjoint source 1 with "
                "shape %s. It must return a one-dimensional array." % str(
                    adjoint_source.shape))
        if len(adjoint_source) != npts:
            raise PyadjointError(
                "The underlying function returned an adjoint source 1 with %i "
                "samples. It must return a function with %i samples which is "
                "the sample count of the input data." % (
                    len(adjoint_source), npts))
        # Make sure the data returned has no infs or NaNs.
        if not np.isfinite(adjoint_source).all():
            raise PyadjointError(
                "The underlying function returned an adjoint source 1 with "
                "either NaNs or Inf values. This must not be.")
    else:
        adjoint_source = None

    return AdjointSource(adj_src_type, misfit=misfit,
                         adjoint_source=adjoint_source,
                         dt=observed1.stats.delta,
                         min_period=config.min_period,
                         max_period=config.max_period,
                         network=observed1.stats.network,
                         station=observed1.stats.station,
                         component=observed1.stats.channel,
                         location=observed1.stats.location,
                         starttime=observed1.stats.starttime)


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

    fct_name = "calculate_adjoint_source"
    name_attr = "VERBOSE_NAME"
    desc_attr = "DESCRIPTION"
    add_attr = "ADDITIONAL_PARAMETERS"

    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        if not hasattr(m, fct_name):
            continue
        fct = getattr(m, fct_name)
        if not callable(fct):
            continue

        name = modname.split('.')[-1]

        if not hasattr(m, name_attr):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, name_attr))

        if not hasattr(m, desc_attr):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, desc_attr))

        # Add tuple of name, verbose name, and description.
        AdjointSource._ad_srcs[name] = (
            fct,
            getattr(m, name_attr),
            getattr(m, desc_attr),
            getattr(m, add_attr) if hasattr(m, add_attr) else None)


_discover_adjoint_sources()


def _discover_adjoint_sources_DD():
    """
    Discovers the available adjoint sources for double difference.
    This should work no matter if
    pyadjoint is checked out from git, packaged as .egg or for any other
    possibility.
    """
    from . import adjoint_source_types

    AdjointSource._ad_srcs_DD = {}

    fct_name = "calculate_adjoint_source_DD"
    name_attr = "VERBOSE_NAME"
    desc_attr = "DESCRIPTION"
    add_attr = "ADDITIONAL_PARAMETERS"

    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types_DD")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        if not hasattr(m, fct_name):
            continue
        fct = getattr(m, fct_name)
        if not callable(fct):
            continue

        name = modname.split('.')[-1]

        if not hasattr(m, name_attr):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, name_attr))

        if not hasattr(m, desc_attr):
            raise PyadjointError(
                "Adjoint source '%s' does not have a variable named %s." %
                (name, desc_attr))

        # Add tuple of name, verbose name, and description.
        AdjointSource._ad_srcs_DD[name] = (
            fct,
            getattr(m, name_attr),
            getattr(m, desc_attr),
            getattr(m, add_attr) if hasattr(m, add_attr) else None)


_discover_adjoint_sources_DD()
