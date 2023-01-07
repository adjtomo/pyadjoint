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

.. code:: python

    # Helper function to get some example data used for
    # illustrative purposes.
    obs, syn = pyadjoint.utils.get_example_data()
    # Select the vertical components of both.
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]


Essentially all of ``Pyadjoint``'s functionality is accessed through its
central :func:`~pyadjoint.main.calculate_adjoint_source` function.
A list of available adjoint source types can be found in
:doc:`index`.


.. code:: python

    adj_src = pyadjoint.calculate_adjoint_source(
        # The type of misfit measurement and adjoint source.
        adj_src_type="waveform_misfit",
        # Pass observed and synthetic data traces.
        observed=obs, synthetic=syn,
        # The spectral content of the data.
        min_period=20.0, max_period=100.0,
        # List of window borders in seconds since the first sample.
        windows=[(800., 900.)]
        )
    print(adj_src)


It returns an :class:`~pyadjoint.adjoint_source.AdjointSource` object.


.. code:: python

    # Access misfit and adjoint sources. The misfit is a floating point number.
    print(adj_src.misfit)
    # The adjoint source is a a numpy array.
    print(adj_src.adjoint_source)

Usage Options
~~~~~~~~~~~~~

All adjoint source types can also be plotted during the calculation. The
type of plot produced depends on the type of misfit measurement and
adjoint source.

.. code:: python

    pyadjoint.calculate_adjoint_source(
        adj_src_type="waveform_misfit", observed=obs, synthetic=syn,
        min_period=20.0, max_period=100.0,
        left_window_border=800.0, right_window_border=900.0, plot=True);


Many types of adjoint sources have additional arguments that can be passed to
it. The waveform misfit adjoint source for example allows to specifiy the
width and type of the taper applied to the data. Please see the documentation
of the different :doc:`index` for details.


.. code:: python

    print(pyadjoint.calculate_adjoint_source(
        adj_src_type="waveform_misfit", observed=obs, synthetic=syn,
        min_period=20.0, max_period=100.0, windows=[(800., 900.)],
        taper_percentage=0.3, taper_type="cosine"))

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
