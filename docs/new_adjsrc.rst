Add a New Adjoint Source
=========================

The following page provides a brief overview on adding new adjoint sources to
this package.

Naming
------

New adjoint sources should be named in snake case and not match an existing
adjoint source type. For example, the instantaneous phase misfit should have

::

    adjsrc_type = "instantaneous_phase"

Double difference adjoint sources must end in '_dd', so a double difference
instantaneous phase misfit should be

::

    adjsrc_type = "instantaneous_phase_dd"

Adjoint Source Type
-------------------

New adjoint sources should be placed in
``pyadjoint.pyadjoint.adjoint_source_types``.

Have a look at existing adjoint sources to learn their code structure, but at
the simplest the new adjoint source file should have a function
``calculate_adjoint_source()``.

.. code:: python

    # Each adjoint source file must contain a calculate_adjoint_source()
    # function. It must take config, observed, synthetic, and windows.
    # Other optional keyword arguments are possible.
    def calculate_adjoint_source(observed, synthetic, config, windows, **kwargs):
        """
        Some information defining the adjoint source
        """
        # This function must return a dictionary
        ret_val = {}

        # >>> calculate `misfit` and `adjoint_source` here

        ret_val["misfit"] = # ...
        ret_val["adjoint_source"] = # ...

        return ret_val

The main processing function expects two keys in the return dictionary,
``misfit`` and ``adjoint_source``. Misfit is a float value representing the
total misfit calculated. Adjoint source should be a NumPy array representing
the calculated adjoint source.

.. note::

    The adjoint source returned by ``calculate_adjoint_source`` **must** be
    time reversed with respect to the synthetic waveform

Double Difference
~~~~~~~~~~~~~~~~~

Double difference adjoint sources require additional inputs and outputs.

.. code:: python

    def calculate_adjoint_source(observed, synthetic, config, windows,
                                 observed_2, synthetic_2, windows_2):
        """
        Some information defining the adjoint source
        """
        ret_val = {}

        # >>> calculate `misfit`, `adjoint_source` and `adjoint_source_2` here

        ret_val["misfit"] = # ...
        ret_val["adjoint_source"] = # ...
        ret_val["adjoint_source_2"] = # ...

        return ret_val

Config
------

You must include your new adjoint source type in the config file.

.. note::
    If your new adjoint source type is a double difference technique, then
    ``adjsrc_type`` must end in ``_dd`` (e.g,. 'waveform_dd'). Pyadjoint will
    recognize this and treat the remaining logic accordingly.

1) Add new ``adjsrc_type`` to the constant list in ``ADJSRC_TYPES``
2) Add new adjoint source to the ``get_config()`` function. If your adjoint source
   requires it's own Config object (due to special parameters), you will need to
   create one.
3) Add new adjoint source to the ``get_function()`` function. This allows
   Pyadjoint to find the correct function based on the name of the adjoint source.