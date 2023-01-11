:py:mod:`pyadjoint.main`
========================

.. py:module:: pyadjoint.main

.. autoapi-nested-parse::

   Main processing scripts to calculate adjoint sources based on two waveforms



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.main.calculate_adjoint_source
   pyadjoint.main.plot_adjoint_source
   pyadjoint.main.get_example_data



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


