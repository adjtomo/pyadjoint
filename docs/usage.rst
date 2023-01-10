Usage
=====

This page illustrates how to calculate misfit and generate adjoint sources
using Pyadjoint. Pyadjoint is mainly dependent on
`ObsPy <https://docs.obspy.org/>`__ for seismic data handling.

.. code:: python

    import obspy
    import pyadjoint

Pyadjoint expects data to be fully preprocessed beforehand. Observed and
synthetic data are expected to have exactly the same length, sampling rate, and
spectral content. It is up to the User to input correctly formatted data.

Pyadjoint is generic and does not care what traces are provided. Despite this,
we will talk about traces as "observed" and "synthetic."

Example Data
~~~~~~~~~~~~

The package comes with example data used for illustrative and debugging
purposes. See the `Example Dataset <example_dataset.html>`__ page for an
explanation of where this data came from.


.. code:: python

    obs, syn = pyadjoint.get_example_data()

    # Pyadjoint requires individual components, not data streams
    # Data are available in R, T and Z components
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

Config
~~~~~~

Each misfit function requires a corresponding :class:`~pyadjoint.config.Config`
class to control optional processing parameters. The
:meth:`~pyadjoint.config.get_config` function provides a wrapper for grabbing
the appropriate Config object.

.. code:: python

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100.)

A list of available adjoint sources can be found at the :doc:`~available` page.


.. code:: python

    >>> print(pyadjoint.discover_adjoint_sources().keys())
    dict_keys(['cc_traveltime_misfit', 'exponentiated_phase_misfit', 'multitaper_misfit', 'waveform_misfit'])


.. note::

    Some of these functions allow for modifiers such as the use of
    double difference measurements. See the 'double difference' section below
    for more explanation

Many types of adjoint sources have additional arguments that can be passed to
it. See the :mod:`~pyadjoint.config` page for available keyword arguments
and descriptions.

.. code:: python

    config = pyadjoint.get_config(adjsrc_type="waveform_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cos")

Calculating Adjoint Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Essentially all of ``Pyadjoint``'s functionality is accessed through its
central :func:`~pyadjoint.main.calculate_adjoint_source` function. This function
takes the previously defined Config class, the observed and synthetic waveforms,
and a list of time windows.

.. code:: python

    adj_src = pyadjoint.calculate_adjoint_source(
        config=config,
        # Pass observed and synthetic data traces.
        observed=obs, synthetic=syn,
        # List of window borders in seconds since the first sample.
        windows=[(800., 900.)]
        )


The function returns an :class:`~pyadjoint.adjoint_source.AdjointSource` object
which has a number of useful attributes for understanding misfit.

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


Time Windows
~~~~~~~~~~~~

Time windows are typically used in misfit quantification to isolate portions
of waveforms that include signals of interest.

Individual time windows represent the start and end time (units: s) of a
window in which to consider waveform misfit, and multiple overlapping time
windows can be included in the final adjoint source.

For example, to include multiple windows:

.. code::

    windows = [(0, 100), (200, 500), (325, 552)]
    adj_src = pyadjoint.calculate_adjoint_source(
        config=config, observed=obs, synthetic=syn, windows=windows
        )

To calculate misfit on the **entire trace**, we need to consider all time steps
in the trace:

.. code::

    windows = [(0, int(obs.stats.npts * obs.stats.dt)]


Double Difference Measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Double difference misfit functions, defined by [Yuan2016]_, construct misfit
and adjoint sources from differential measurements between stations to reduce
the influence of systematic errors from source and stations. 'Differential' is
defined as "between pairs of stations, from a common source."

Double difference measurements require a second set of observed and synthetic
waveforms, as well as windows for this second set of waveforms. The new
windows can be independent of the first set of windows, but must contain
the same number of windows. Each window will be compared in order.

The ``choice`` parameter in the :func:`~pyadjoint.main.calculate_adjoint_source`
function signals to Pyadjoint modifications to the input arguments may be
required. See individual adjoint sources for their respective usage.

.. note::

    In the following code snippet, we use the 'R' component of the same station
    in lieu of waveforms from a second station. In practice, the second set of
    waveforms should come from a completely different station.

.. code:: python

    obs_2, syn_2 = pyadjoint.get_example_data()
    obs_2 = obs_2.select(component="R")[0]
    syn_2 = syn_2.select(component="R")[0]

    adj_src, adj_src_2 = pyadjoint.calculate_adjoint_source(
        config=config, observed=obs, synthetic=syn, windows=[(800., 900.)]
        choice="double_difference",
        observed_2=obs_2, synthetic_2,syn_2, windows_2=[(800., 900.,)]
        )

Double difference misfit functions result in two adjoint sources, one for each
station in the pair of waveforms.

Plotting Adjoint Sources
~~~~~~~~~~~~~~~~~~~~~~~~

All adjoint source types can be plotted during calculation. The
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
