:py:mod:`pyadjoint`
===================

.. py:module:: pyadjoint

.. autoapi-nested-parse::

   :copyright:
       adjTomo Dev Team (adjtomo@gmail.com), 2022
       Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   adjoint_source_types/index.rst
   tests/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   adjoint_source/index.rst
   config/index.rst
   main/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pyadjoint.AdjointSource



Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.calculate_adjoint_source
   pyadjoint.get_example_data
   pyadjoint.plot_adjoint_source
   pyadjoint.get_config
   pyadjoint.get_function



Attributes
~~~~~~~~~~

.. autoapisummary::

   pyadjoint.__version__
   pyadjoint.logger
   pyadjoint.propagate
   pyadjoint.ch
   pyadjoint.FORMAT
   pyadjoint.formatter
   pyadjoint.ADJSRC_TYPES


.. py:data:: __version__
   :annotation: = 0.1.0

   

.. py:exception:: PyadjointError

   Bases: :py:obj:`Exception`

   Base class for all Pyadjoint exceptions. Will probably be used for all
   exceptions to not overcomplicate things as the whole package is pretty
   small.


.. py:exception:: PyadjointWarning

   Bases: :py:obj:`UserWarning`

   Base class for all Pyadjoint warnings.


.. py:data:: logger
   

   

.. py:data:: propagate
   :annotation: = 0

   

.. py:data:: ch
   

   

.. py:data:: FORMAT
   :annotation: = [%(asctime)s] - %(name)s - %(levelname)s: %(message)s

   

.. py:data:: formatter
   

   

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



.. py:function:: calculate_adjoint_source(observed, synthetic, config, windows, plot=False, plot_filename=None, observed_2=None, synthetic_2=None, windows_2=None, **kwargs)

   Central function of Pyadjoint used to calculate adjoint sources and misfit.

   This function uses the notion of observed and synthetic data to offer a
   nomenclature most users are familiar with. Please note that it is
   nonetheless independent of what the two data arrays actually represent.

   The function tapers the data from ``left_window_border`` to
   ``right_window_border``, both in seconds since the first sample in the
   data arrays.

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


.. py:function:: get_example_data()

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


.. py:function:: plot_adjoint_source(observed, synthetic, adjoint_source, misfit, windows, adjoint_source_name)

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


.. py:function:: get_config(adjsrc_type, min_period, max_period, **kwargs)

   Defines two common parameters for all configuration objects and then
   reassigns self to a sub Config class which dictates its own required
   parameters


.. py:function:: get_function(adjsrc_type)

   Wrapper for getting the correct adjoint source function based on the
   `adjsrc_type`. Many adjoint sources share functions with different flags
   so this function takes care of the logic of choosing which.

   :type adjsrc_type: str
   :param adjsrc_type: choice of adjoint source
   :rtype: function
   :return: calculate_adjoint_source function for the correct adjoint source
       type


.. py:data:: ADJSRC_TYPES
   

   

