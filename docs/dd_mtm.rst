Double Difference Multitaper Misfit
===================================

.. note::

    Double difference misfit functions, defined in [Yuan2016]_, construct misfit
    and adjoint sources from differential measurements between stations to reduce
    the influence of systematic errors from source and stations. 'Differential' is
    defined as "between pairs of stations, from a common source".

Due to the length and complexity of the equations for double difference
multitaper misfit, please see [Yuan2016]_ Appendix A1 and A2 for the
mathematical expressions that define misfit and adjoint source.


Usage
`````

The following code snippets illustrates the basic usage of the double
difference multitaper misfit function.

Note that double difference implementations can take a set of windows for the
second set of waveforms, independent of the first set of windows. Windows
are compared in order, so both ``windows`` and ``windows_2`` need to be the same
length.

.. note::

    In the following code snippet, we use the 'R' component of the same station
    in liue of waveforms from a second station. In practice, the second set of
    waveforms should come from a completely different station.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    obs_2, syn_2 = pyadjoint.get_example_data()
    obs_2 = obs_2.select(component="R")[0]
    syn_2 = syn_2.select(component="R")[0]

    config = pyadjoint.get_config(adjsrc_type="multitaper_misfit", min_period=20.,
                                  max_period=100., taper_percentage=0.3,
                                  taper_type="cos")

    # Calculating double-difference adjoint source returns two adjoint sources
    adj_src, adj_src_2 = pyadjoint.calculate_adjoint_source(
        config=config, observed=obs, synthetic=syn, windows=[(800., 900.)],
        choice="double_difference", observed_2=obs_2, synthetic_2=syn_2,
        windows_2=[(800., 900.)]
        )

