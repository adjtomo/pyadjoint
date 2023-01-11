:py:mod:`pyadjoint.adjoint_source_types.cc_traveltime_misfit`
=============================================================

.. py:module:: pyadjoint.adjoint_source_types.cc_traveltime_misfit

.. autoapi-nested-parse::

   Cross correlation traveltime misfit and associated adjoint source.

   :authors:
       adjtomo Dev Team (adjtomo@gmail.com), 2023
       Yanhua O. Yuan (yanhuay@princeton.edu), 2017
       Youyi Ruan (youyir@princeton.edu) 2016
       Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.adjoint_source_types.cc_traveltime_misfit.calculate_adjoint_source



.. py:function:: calculate_adjoint_source(observed, synthetic, config, windows, observed_2=None, synthetic_2=None, windows_2=None)

   Calculate adjoint source for the cross-correlation traveltime misfit
   measurement

   :type observed: obspy.core.trace.Trace
   :param observed: observed waveform to calculate adjoint source
   :type synthetic:  obspy.core.trace.Trace
   :param synthetic: synthetic waveform to calculate adjoint source
   :type config: pyadjoint.config.ConfigCCTraveltime
   :param config: Config class with parameters to control processing
   :type windows: list of tuples
   :param windows: [(left, right),...] representing left and right window
       borders to be used to calculate misfit and adjoint sources
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


