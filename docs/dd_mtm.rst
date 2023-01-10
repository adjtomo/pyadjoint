Double Difference Multitaper Misfit
===================================

Due to the length and complexity of the equations for double difference
multitaper misfit, please see [Yuan2016]_
(https://academic.oup.com/gji/article/206/3/1599/2583519) Appendix sections A1
and A2 for the mathematical derivation and expresssions that define misfit and
adjoint source.


Usage
`````

The following code snippet illustrates the basic usage of the double
difference multitaper misfit function.

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

