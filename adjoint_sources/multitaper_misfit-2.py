import pyadjoint
import matplotlib.pylab as plt
fig = plt.figure(figsize=(12, 7))
obs, syn = pyadjoint.utils.get_example_data()
obs = obs.select(component="T")[0]
syn = syn.select(component="T")[0]
start, end = pyadjoint.utils.EXAMPLE_DATA_SDIFF
pyadjoint.calculate_adjoint_source("multitaper_misfit", obs, syn, 20.0, 100.0,
                                   start, end, adjoint_src=True, plot=fig)
plt.show()