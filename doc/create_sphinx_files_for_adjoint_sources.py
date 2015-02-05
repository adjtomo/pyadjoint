#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
This will create the sphinx input files for the various defined adjoint
sources.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import os


folder = "adjoint_sources"
if not os.path.exists(folder):
    os.makedirs(folder)


import pyadjoint


TEMPLATE = """
{upper}
{name}
{lower}

{description}

Example Plots
-------------

Pdiff Phase on Vertical Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::

    import pyadjoint
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(12, 6))
    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]
    start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
    pyadjoint.calculate_adjoint_source("{short_name}", obs, syn, 20.0, 100.0,
                                       start, end, adjoint_src=True, plot=fig)
    plt.show()


Sdiff Phase on Transverse Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::

    import pyadjoint
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(12, 6))
    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="T")[0]
    syn = syn.select(component="T")[0]
    start, end = pyadjoint.utils.EXAMPLE_DATA_SDIFF
    pyadjoint.calculate_adjoint_source("{short_name}", obs, syn, 20.0, 100.0,
                                       start, end, adjoint_src=True, plot=fig)
    plt.show()

""".lstrip()

srcs = pyadjoint.AdjointSource._ad_srcs

srcs = [(key, value) for key, value in srcs.items()]
srcs = sorted(srcs, key=lambda x: x[1][1])

for key, value in srcs:
    filename = os.path.join(folder, "%s.rst" % key)

    with open(filename, "wt") as fh:
        fh.write(TEMPLATE.format(
            upper="=" * len(value[1].strip()),
            name=value[1].strip(),
            lower="=" * len(value[1].strip()),
            description=value[2].lstrip(),
            short_name=key
        ))

INDEX = """
===============
Adjoint Sources
===============

``Pyadjoint`` currently contains the following adjoint sources:
""".strip()

index_filename = os.path.join(folder, "index.rst")
with open(index_filename, "wt") as fh:
    fh.write(INDEX)
    fh.write("\n\n\n")
    fh.write("\n".join([":doc:`%s`" % _i[0] for _i in srcs]))
