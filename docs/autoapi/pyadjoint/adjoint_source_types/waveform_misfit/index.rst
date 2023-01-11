:py:mod:`pyadjoint.adjoint_source_types.waveform_misfit`
========================================================

.. py:module:: pyadjoint.adjoint_source_types.waveform_misfit

.. autoapi-nested-parse::

   Simple waveform-based misfit and adjoint source. Contains options for
   'waveform' and 'convolution' misfit functions, as well as their double
   difference counterparts

   This one function takes care of implementation for multiple misfit functions:
       - 'waveform': Waveform misfit from [Tromp2005]_
       - 'convolution': Waveform convolution misfit from [Choi2011]_
       - 'waveform_dd': Double difference waveform misfit from [Yuan2016]_
       - 'convolution_dd': Double difference convolution misfit

   :authors:
       adjTomo Dev Team (adjtomo@gmail.com), 2023
       Yanhua O. Yuan (yanhuay@princeton.edu), 2017
       Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.adjoint_source_types.waveform_misfit.calculate_adjoint_source



.. py:function:: calculate_adjoint_source(observed, synthetic, config, windows, observed_2=None, synthetic_2=None, windows_2=None)

   Calculate adjoint source for the waveform-based misfit measurements.

   :type observed: obspy.core.trace.Trace
   :param observed: observed waveform to calculate adjoint source
   :type synthetic:  obspy.core.trace.Trace
   :param synthetic: synthetic waveform to calculate adjoint source
   :type config: pyadjoint.config.ConfigWaveform
   :param config: Config class with parameters to control processing
   :type windows: list of tuples
   :param windows: [(left, right),...] representing left and right window
       borders to be tapered in units of seconds since first sample in data
       array
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


