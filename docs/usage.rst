Usage
=====

The first step is to import ObsPy and ``Pyadjoint``.

.. code:: python

    import obspy
    import pyadjoint

``Pyadjoint`` expects the data to be fully preprocessed thus both
observed and synthetic data are expected to have exactly the same
length, sampling rate, and spectral content.

``Pyadjoint`` does not care about the actual components in question; it will
use two traces and calculate misfit values and adjoint sources for them. To
provide a familiar nomenclature we will always talk about observed and
synthetic data even though the workflow is independent of what the data
represents.

Example Data
~~~~~~~~~~~~

The package comes with a helper function to get some example data used for
illustrative and debugging purposes.


.. code:: python

    obs, syn = pyadjoint.get_example_data()
    # Select the vertical components of both.
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

Config
~~~~~~

Each misfit function requires a corresponding :class:`~pyadjoint.config.Config`
class to control optional processing parameters. The
:meth:`~pyadjoint.config.get_config` function provides a wrapper for grabbing
the appropriate Config.

.. code:: python

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100.)

A list of available adjoint source types can be found using the
:meth:`~pyadjoint.discover_adjoint_sources` function.

.. code:: python

    >>> print(pyadjoint.discover_adjoint_sources().keys())
    dict_keys(['cc_traveltime_misfit', 'exponentiated_phase_misfit', 'multitaper_misfit', 'waveform_misfit'])


.. note::

    Some of these functions allow for modifiers such as the use of
    double difference measurements. See individual misfit function pages for
    options.

Many types of adjoint sources have additional arguments that can be passed to
it.

.. code:: python

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cosine")

Calculate Adjoint Source
~~~~~~~~~~~~~~~~~~~~~~~~

Essentially all of ``Pyadjoint``'s functionality is accessed through its
central :func:`~pyadjoint.main.calculate_adjoint_source` function.


.. code:: python

    adj_src = pyadjoint.calculate_adjoint_source(
        config=config,
        # Pass observed and synthetic data traces.
        observed=obs, synthetic=syn,
        # List of window borders in seconds since the first sample.
        windows=[(800., 900.)]
        )

The function returns an :class:`~pyadjoint.adjoint_source.AdjointSource` object.

.. code::

    >>> print(adj_src)
    'waveform_misfit' Adjoint Source for channel MXZ at station SY.DBO
        misfit: 4.263e-11
        adjoint_source: available with 3600 samples
        windows: generated with 1 windows
    # Access misfit and adjoint sources. The misfit is a floating point number.
    >>> print(adj_src.misfit)
    4.263067857359352e-11
    # The adjoint source is a a numpy array.
    >>> print(adj_src.adjoint_source)
    [0. 0. 0. ... 0. 0. 0.]
    # Time windows used to generate the array are stored
    >>> print(adj_src.windows)
    [(800.0, 900.0)]
    # Misfit stats for each window are also stored
    >>> print(adj_src.window_stats)
    [{'type': 'waveform', 'left': 800.0, 'right': 901.0, 'misfit': 4.263067857359352e-11, 'difference': 1.519230269510467e-08}]

Plotting Adjoint Sources
~~~~~~~~~~~~~~~~~~~~~~~~

All adjoint source types can also be plotted during the calculation. The
type of plot produced depends on the type of misfit measurement and
adjoint source.

.. code:: python

    pyadjoint.calculate_adjoint_source(config=config, observed=obs,
                                       synthetic=syn, plot=True,
                                       plot_filename="./waveform_adjsrc.png");


Saving to Disk
~~~~~~~~~~~~~~

One of course wants to serialize the calculated adjoint sources to disc at one
point in time. You need to pass the filename and the desired format as well as
some format specific parameters to the
:meth:`~pyadjoint.adjoint_source.AdjointSource.write` method of the
:class:`~pyadjoint.adjoint_source.AdjointSource` object. Instead of a filename
you can also pass an open file or a file-like object. Please refer to its
documentation for more details.


.. code:: python

    adj_src.write(filename="NET.STA.CHA.adj_src",
                  format="SPECFEM", time_offset=-10)
