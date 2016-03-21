import pyadjoint
# Yanhua O. Yuan
adj_src = "instantaneous_phase_misfit"

config = pyadjoint.Config(min_period=30.0, max_period=75.0,
                          lnpt=15,
                          transfunc_waterlevel=1.0E-10,
                          water_threshold=0.02,
                          ipower_costaper=10,
                          min_cycle_in_window=3,
                          taper_percentage=0.3,
                          taper_type='hann',
                          mt_nw=4,
                          phase_step=1.5,
                          use_cc_error=False,
                          use_mt_error=False)

obs, syn = pyadjoint.utils.get_example_data()
obs = obs.select(component="Z")[0]
syn = syn.select(component="Z")[0]

window = [[2076., 2418.0]]

a_src = pyadjoint.calculate_adjoint_source(
        adj_src_type=adj_src, observed=obs, synthetic=syn,
        config=config, window=window,
        adjoint_src=True, plot=False)
