Double Difference Multitaper Misfit
===================================

Due to the length and complexity of the equations for double difference
multitaper misfit, please see [Yuan2016]_
(`Link to paper <https://academic.oup.com/gji/article/206/3/1599/2583519>`__)
Appendix sections A1 and A2 for the mathematical derivation and expresssions
that define misfit and adjoint source.


Usage
`````

::

    adjsrc_type = "multitaper_dd"

The following code snippet illustrates the basic usage of the cross correlation
traveltime misfit function.  See the corresponding
`Config <autoapi/pyadjoint/config/index.html#pyadjoint.config.ConfigMultitaper>`__
object for additional configuration parameters.

.. code:: python

    import pyadjoint

    obs, syn = pyadjoint.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    obs_2, syn_2 = pyadjoint.get_example_data()
    obs_2 = obs_2.select(component="R")[0]
    syn_2 = syn_2.select(component="R")[0]

    config = pyadjoint.get_config(adjsrc_type="multitaper_dd", min_period=20.,
                                  max_period=100.)

    # Calculating double-difference adjoint source returns two adjoint sources
    adj_src, adj_src_2 = pyadjoint.calculate_adjoint_source(
        config=config, observed=obs, synthetic=syn, windows=[(800., 900.)],
        observed_2=obs_2, synthetic_2=syn_2, windows_2=[(800., 900.)]
        )

