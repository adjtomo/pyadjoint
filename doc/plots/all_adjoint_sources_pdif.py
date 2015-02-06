import pyadjoint
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

srcs = pyadjoint.AdjointSource._ad_srcs

srcs = [(key, value) for key, value in srcs.items()]
srcs = sorted(srcs, key=lambda x: x[1][1])

plt.figure(figsize=(12, 2 * (len(srcs) + 1) ))


observed, synthetic = pyadjoint.utils.get_example_data()

obs = observed.select(component="Z")[0]
syn = synthetic.select(component="Z")[0]

start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF

left_window_border,right_window_border = start, end
x_range = obs.stats.endtime - obs.stats.starttime
buf = (right_window_border - left_window_border) * 1.0
left_window_border -= buf
right_window_border += buf
left_window_border = max(0, left_window_border)
right_window_border = min(x_range, right_window_border)

ylim = np.abs(obs.slice(start, x_range - end).data).max() * 1.15

plt.subplot(len(srcs) + 1, 1, 1)
plt.plot(obs.times(), obs.data, color="0.2", label="observed", lw=2)
plt.plot(syn.times(), syn.data, color="#B26063", label="synthetic", lw=2)
plt.legend(fancybox=True, framealpha=0.5)
plt.grid()

plt.ylim(-ylim, ylim)
plt.xlim(left_window_border, right_window_border)

re = Rectangle((start, plt.ylim()[0]), end - start,
               plt.ylim()[1] - plt.ylim()[0], color="blue",
               alpha=0.25, zorder=-5)
plt.gca().add_patch(re)
plt.text(x=end - 0.02 * (end - start),
         y=plt.ylim()[1] - 0.01 * (plt.ylim()[1] - plt.ylim()[0]),
         s="Chosen window",
         color="0.2",
         fontweight=900,
         horizontalalignment="right",
         verticalalignment="top",
         size="small", multialignment="right")


pretty_colors = ["#5B76A1", "#619C6F", "#867CA8", "#BFB281", "#74ACBD"]

_i = 0
for key, value in srcs:
    _i += 1
    plt.subplot(len(srcs) + 1, 1, _i + 1)

    adj_src = pyadjoint.calculate_adjoint_source(key, obs, syn, 20, 100,
                                                 start, end, adjoint_src=True)

    plt.plot(obs.times(), adj_src.adjoint_source,
             color=pretty_colors[(_i - 1) % len(pretty_colors)], lw=2,
             label=value[1])
    plt.xlim(x_range - right_window_border, x_range - left_window_border)
    plt.legend(fancybox=True, framealpha=0.5)
    plt.grid()

plt.tight_layout()
plt.show()
