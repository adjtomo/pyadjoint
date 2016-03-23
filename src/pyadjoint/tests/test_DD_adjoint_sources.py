import pyadjoint
# Yanhua O. Yuan
adj_src = "cc_traveltime_misfit_DD"
adj_src = "multitaper_misfit_DD"
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

window1 = [[1993., 2063.0]]
window2 = [[2193., 2263.0]]

a_src = pyadjoint.calculate_adjoint_source_DD(
        adj_src_type=adj_src, observed1=obs,
        synthetic1=syn, observed2=obs, synthetic2=syn,
        config=config, window1=window1, window2=window2,
        adjoint_src=True, plot=True)
