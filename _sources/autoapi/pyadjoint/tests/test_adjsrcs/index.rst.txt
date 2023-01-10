:py:mod:`pyadjoint.tests.test_adjsrcs`
======================================

.. py:module:: pyadjoint.tests.test_adjsrcs

.. autoapi-nested-parse::

   Test generalized adjoint source generation for each type



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.tests.test_adjsrcs.example_data
   pyadjoint.tests.test_adjsrcs.example_2_data
   pyadjoint.tests.test_adjsrcs.example_window
   pyadjoint.tests.test_adjsrcs.test_waveform_misfit
   pyadjoint.tests.test_adjsrcs.test_waveform_dd_misfit
   pyadjoint.tests.test_adjsrcs.test_convolved_waveform_misfit
   pyadjoint.tests.test_adjsrcs.test_dd_convolved_waveform_misfit
   pyadjoint.tests.test_adjsrcs.test_cc_traveltime_misfit
   pyadjoint.tests.test_adjsrcs.test_dd_cc_traveltime_misfit
   pyadjoint.tests.test_adjsrcs.test_multitaper_misfit
   pyadjoint.tests.test_adjsrcs.test_dd_multitaper_misfit
   pyadjoint.tests.test_adjsrcs.test_exponentiated_phase_misfit



Attributes
~~~~~~~~~~

.. autoapisummary::

   pyadjoint.tests.test_adjsrcs.PLOT
   pyadjoint.tests.test_adjsrcs.path


.. py:data:: PLOT
   :annotation: = False

   

.. py:data:: path
   :annotation: = ./

   

.. py:function:: example_data()

   Return example data to be used to test adjoint sources


.. py:function:: example_2_data()

   Return example data to be used to test adjoint source double difference
   calculations. Simply grabs the R component to provide a different waveform


.. py:function:: example_window()

   Defines an example window where misfit can be quantified


.. py:function:: test_waveform_misfit(example_data, example_window)

   Test the waveform misfit function


.. py:function:: test_waveform_dd_misfit(example_data, example_2_data, example_window)

   Test the waveform misfit function


.. py:function:: test_convolved_waveform_misfit(example_data, example_window)

   Test the waveform misfit function


.. py:function:: test_dd_convolved_waveform_misfit(example_data, example_2_data, example_window)

   Test the waveform misfit function


.. py:function:: test_cc_traveltime_misfit(example_data, example_window)

   Test the waveform misfit function


.. py:function:: test_dd_cc_traveltime_misfit(example_data, example_2_data, example_window)

   Test the waveform misfit function


.. py:function:: test_multitaper_misfit(example_data, example_window)

   Test the waveform misfit function


.. py:function:: test_dd_multitaper_misfit(example_data, example_2_data, example_window)

   Test the waveform misfit function


.. py:function:: test_exponentiated_phase_misfit(example_data, example_window)

   Test the waveform misfit function


