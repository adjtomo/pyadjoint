#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Simple L2 norm adjoint source.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""

NAME = "l2_norm"

DESCRIPTION = """
Some longer description
"""


def calculate_misfit(observed, synthetic, min_period, max_period):
    pass


def calculate_adjoint_source(observed, synthetic, min_period, max_period,
                             left_window_border, right_window_border,
                             adjoint_src, figure, **kwargs):
    ret_val = {}

    diff = synthetic - observed
    ret_val["misfit"] = (diff ** 2).sum()

    if adjoint_src is True:
        ret_val["adjoint_source"] = (-1.0 * diff)[::-1]

    if figure:
        pass

    return ret_val
