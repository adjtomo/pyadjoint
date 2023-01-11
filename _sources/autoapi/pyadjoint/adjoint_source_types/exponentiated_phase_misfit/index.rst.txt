:py:mod:`pyadjoint.adjoint_source_types.exponentiated_phase_misfit`
===================================================================

.. py:module:: pyadjoint.adjoint_source_types.exponentiated_phase_misfit

.. autoapi-nested-parse::

   Exponentiated phase misfit function and adjoint source.

   :authors:
       adjtomo Dev Team (adjtomo@gmail.com), 2022
       Yanhua O. Yuan (yanhuay@princeton.edu), 2016
   :license:
       BSD 3-Clause ("BSD New" or "BSD Simplified")



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.adjoint_source_types.exponentiated_phase_misfit.calculate_adjoint_source



.. py:function:: calculate_adjoint_source(observed, synthetic, config, windows, adjoint_src=True, window_stats=True, **kwargs)

   Calculate adjoint source for the exponentiated phase misfit.

   :type observed: obspy.core.trace.Trace
   :param observed: observed waveform to calculate adjoint source
   :type synthetic:  obspy.core.trace.Trace
   :param synthetic: synthetic waveform to calculate adjoint source
   :type config: pyadjoint.config.ConfigExponentiatedPhase
   :param config: Config class with parameters to control processing
   :type windows: list of tuples
   :param windows: [(left, right),...] representing left and right window
       borders to be used to calculate misfit and adjoint sources
   :type adjoint_src: bool
   :param adjoint_src: flag to calculate adjoint source, if False, will only
       calculate misfit
   :type window_stats: bool
   :param window_stats: flag to return stats for individual misfit windows used
       to generate the adjoint source


