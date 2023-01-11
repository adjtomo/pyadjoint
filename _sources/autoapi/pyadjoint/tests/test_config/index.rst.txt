:py:mod:`pyadjoint.tests.test_config`
=====================================

.. py:module:: pyadjoint.tests.test_config

.. autoapi-nested-parse::

   Test suite for Config class



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pyadjoint.tests.test_config.test_all_configs
   pyadjoint.tests.test_config.test_config_set_correctly
   pyadjoint.tests.test_config.test_incorrect_inputs
   pyadjoint.tests.test_config.test_adjsrc_types
   pyadjoint.tests.test_config.test_get_functions



.. py:function:: test_all_configs()

   Test importing all configs based on available types


.. py:function:: test_config_set_correctly()

   Just make sure that choosing a specific adjoint source type exposes
   the correct parameters


.. py:function:: test_incorrect_inputs()

   Make sure that incorrect input parameters are error'd


.. py:function:: test_adjsrc_types()

   Check that all adjoint sources have the correct format


.. py:function:: test_get_functions()

   Get correct functions based on adjoint source types


