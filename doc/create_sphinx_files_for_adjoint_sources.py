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
""".strip()

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
            description=value[2].strip()
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
