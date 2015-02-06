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

{additional_parameters}

Usage
-----

.. doctest::

    >>> import pyadjoint
    >>> obs, syn = pyadjoint.utils.get_example_data()
    >>> obs = obs.select(component="Z")[0]
    >>> syn = syn.select(component="Z")[0]
    >>> start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
    >>> adj_src = pyadjoint.calculate_adjoint_source(
    ...     adj_src_type="{short_name}", observed=obs, synthetic=syn,
    ...     min_period=20.0, max_period=100.0, left_window_border=start,
    ...     right_window_border=end)
    >>> print(adj_src)
    {name} Adjoint Source for component Z at station SY.DBO
        Misfit: 4.26e-11
        Adjoint source available with 3600 samples

Example Plots
-------------

The following shows plots of the :doc:`../example_dataset` for some phases.

Pdif Phase on Vertical Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example contains *Pdif* and some surface reflected diffracted phases
recorded on the vertical component.

.. plot::

    import pyadjoint
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(12, 7))
    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]
    start, end = pyadjoint.utils.EXAMPLE_DATA_PDIFF
    pyadjoint.calculate_adjoint_source("{short_name}", obs, syn, 20.0, 100.0,
                                       start, end, adjoint_src=True, plot=fig)
    plt.show()


Sdif Phase on Transverse Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example contains *Sdif* and some surface reflected diffracted phases
recorded on the transverse component.

.. plot::

    import pyadjoint
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(12, 7))
    obs, syn = pyadjoint.utils.get_example_data()
    obs = obs.select(component="T")[0]
    syn = syn.select(component="T")[0]
    start, end = pyadjoint.utils.EXAMPLE_DATA_SDIFF
    pyadjoint.calculate_adjoint_source("{short_name}", obs, syn, 20.0, 100.0,
                                       start, end, adjoint_src=True, plot=fig)
    plt.show()
""".lstrip()



ADDITIONAL_PARAMETERS_TEMPLATE = """
Additional Parameters
---------------------

Additional parameters in addition to the default ones in the central
:func:`~pyadjoint.adjoint_source.calculate_adjoint_source` function:

{params}
""".strip()


srcs = pyadjoint.AdjointSource._ad_srcs

srcs = [(key, value) for key, value in srcs.items()]
srcs = sorted(srcs, key=lambda x: x[1][1])

for key, value in srcs:
    filename = os.path.join(folder, "%s.rst" % key)

    additional_params = ""
    if value[3]:
        additional_params = ADDITIONAL_PARAMETERS_TEMPLATE.format(
            params=value[3])

    with open(filename, "wt") as fh:
        fh.write(TEMPLATE.format(
            upper="=" * len(value[1].strip()),
            name=value[1].strip(),
            lower="=" * len(value[1].strip()),
            description=value[2].lstrip(),
            short_name=key,
            additional_parameters=additional_params
        ))

INDEX = """
===============
Adjoint Sources
===============

``Pyadjoint`` can currently calculate the following misfits measurements and
associated adjoint sources:

.. toctree::
    :maxdepth: 1

    {contents}

Comparative Plots of All Available Adjoint Sources
--------------------------------------------------

Pdif Phase on Vertical Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example contains *Pdif* and some surface reflected diffracted phases
recorded on the vertical component.

.. plot:: plots/all_adjoint_sources_pdif.py

Sdif Phase on Transverse Component
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example contains *Sdif* and some surface reflected diffracted phases
recorded on the transverse component.

.. plot:: plots/all_adjoint_sources_sdif.py

""".lstrip()

index_filename = os.path.join(folder, "index.rst")
with open(index_filename, "wt") as fh:
    fh.write(INDEX.format(
        contents="\n    ".join([_i[0] for _i in srcs])))
