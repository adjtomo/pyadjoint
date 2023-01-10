#!/usr/bin/env python3
"""
Central interfaces for ``Pyadjoint``, misfit measurement package.

:copyright:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import numpy as np
import obspy

from pyadjoint import discover_adjoint_sources


class AdjointSource:
    """
    Adjoint Source class to hold calculated adjoint sources
    """
    def __init__(self, adjsrc_type, misfit, dt, min_period, max_period,
                 component, adjoint_source=None, windows=None,
                 network=None, station=None, location=None, starttime=None,
                 window_stats=None):
        """
        Class representing an already calculated adjoint source.

        :param adjsrc_type: The type of adjoint source.
        :type adjsrc_type:  str
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
        :type windows: list of dict
        :param windows: measurement windows used to generate the adjoint
            source, each carrying information about the misfit of the window
        :param network: The network code of the station.
        :type network: str
        :param station: The station code of the station.
        :type station: str
        :param location: The location code of the station.
        :type location: str
        :param starttime: starttime of adjoint source
        :type starttime: obspy.UTCDateTime
        """
        adj_srcs = discover_adjoint_sources()
        if adjsrc_type not in adj_srcs.keys():
            raise ValueError(f"Unknown adjoint source type {adjsrc_type}")

        self.adjsrc_type = adjsrc_type
        self.adj_src_name = adjsrc_type
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
        self.windows = windows
        self.window_stats = window_stats

    def __str__(self):
        if self.network and self.station:
            station = f" at station {self.network}.{self.station}"
        else:
            station = ""

        if self.adjoint_source is not None:
            adj_src_status = \
                f"available with {len(self.adjoint_source)} samples"
        else:
            adj_src_status = "has not been calculated"

        if self.windows is not None:
            windows_status = f"generated with {len(self.windows)} windows"
        else:
            windows_status = "has no windows"

        return (f"'{self.adj_src_name}' Adjoint Source for "
                f"channel {self.component}{station}\n"
                f"\tmisfit: {self.misfit:.4g}\n"
                f"\tadjoint_source: {adj_src_status}\n"
                f"\twindows: {windows_status}"
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

        .. rubric ASDF

        Adjoint sources can be written directly to an ASDFDataSet if provided.
        Optional ``coordinates`` parameter specifies the location of the
        station that generated the adjoint source

        >>> adj_src.write(ds, format="ASDF", time_offset=-10,
        ...               coordinates={'latitude':19.2,
        ...                            'longitude':13.4,
        ...                            'elevation_in_m':2.0})
        """
        if self.adjoint_source is None:
            raise ValueError("Can only write adjoint sources if the adjoint "
                             "source has been calculated.")

        format = format.upper()
        available_formats = ["SPECFEM", "ASDF"]
        if format not in available_formats:
            raise ValueError("format '%s' not known. Available formats: %s" %
                             (format, ", ".join(available_formats)))

        if format == "SPECFEM":
            self._write_specfem(filename, **kwargs)
        elif format == "ASDF":
            self._write

    def _write_specfem(self, filename, time_offset):
        """
        Write the adjoint source for SPECFEM.

        :param filename: name of file to write adjoint source to
        :type filename: str
        :param time_offset: time offset of the first time point in array
        :type time_offset: float
        """
        l = len(self.adjoint_source)

        to_write = np.empty((l, 2))

        to_write[:, 0] = np.linspace(0, (l - 1) * self.dt, l)
        to_write[:, 0] += time_offset
        # SPECFEM expects non-time reversed adjoint sources.
        to_write[:, 1] = self.adjoint_source[::-1]

        np.savetxt(filename, to_write)

    def _write_asdf(self, ds, time_offset, coordinates=None, **kwargs):
        """
        Writes the adjoint source to an ASDF file.

        :param ds: The ASDF data structure read in using pyasdf.
        :type ds: str
        :param time_offset: The temporal offset of the first sample in seconds.
            This is required if using the adjoint source as input to SPECFEM.
        :type time_offset: float
        :param coordinates: If given, the coordinates of the adjoint source.
            The 'latitude', 'longitude', and 'elevation_in_m' of the adjoint
            source must be defined.
        :type coordinates: list
        """
        # Import here to not have a global dependency on pyasdf
        from pyasdf.exceptions import NoStationXMLForStation

        # Convert the adjoint source to SPECFEM format
        l = len(self.adjoint_source)
        specfem_adj_source = np.empty((l, 2))
        specfem_adj_source[:, 0] = np.linspace(0, (l - 1) * self.dt, l)
        specfem_adj_source[:, 0] += time_offset
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
                      "adjoint_source_type": self.adjsrc_type,
                      "min_period": min_period, "max_period": max_period,
                      "latitude": latitude, "longitude": longitude,
                      "elevation_in_m": elevation_in_m,
                      "station_id": station_id, "component": component,
                      "units": "m"}

        # Use pyasdf to add auxiliary data to the ASDF file
        ds.add_auxiliary_data(data=specfem_adj_source,
                              data_type="AdjointSource", path=tag,
                              parameters=parameters)

