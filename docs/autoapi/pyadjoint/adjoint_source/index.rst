:py:mod:`pyadjoint.adjoint_source`
==================================

.. py:module:: pyadjoint.adjoint_source

.. autoapi-nested-parse::

   Central interfaces for ``Pyadjoint``, misfit measurement package.

   :copyright:
       adjTomo Dev Team (adjtomo@gmail.com), 2022
       Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pyadjoint.adjoint_source.AdjointSource




.. py:class:: AdjointSource(adjsrc_type, misfit, dt, min_period, max_period, component, adjoint_source=None, windows=None, network=None, station=None, location=None, starttime=None, window_stats=None)

   Adjoint Source class to hold calculated adjoint sources

   .. py:method:: __str__()

      Return str(self).


   .. py:method:: write(filename, format, **kwargs)

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


   .. py:method:: _write_specfem(filename, time_offset)

      Write the adjoint source for SPECFEM.

      :param filename: name of file to write adjoint source to
      :type filename: str
      :param time_offset: time offset of the first time point in array
      :type time_offset: float


   .. py:method:: _write_asdf(ds, time_offset, coordinates=None, **kwargs)

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



