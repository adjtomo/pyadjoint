#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Tests for the utility functions.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import obspy

from pyadjoint.utils import taper_window


def test_taper_within_window_simple():
    """
    Tests the taper_window function with a silly but simple test.
    """
    tr = obspy.read()[0]
    tr.data = np.require(tr.data, dtype=np.float64)

    tr2 = tr.copy().taper(max_percentage=0.15, type="cosine")

    # Just taper everything
    taper_window(tr, 0, tr.stats.endtime - tr.stats.starttime,
                 taper_percentage=0.15, taper_type="cosine")

    # Deleting the processing information as this will not be the same.
    del tr.stats.processing
    del tr2.stats.processing
    assert tr == tr2


def test_taper_within_window():
    """
    Tests the taper_window function.
    """
    tr = obspy.read()[0]
    length = tr.stats.endtime - tr.stats.starttime

    # Not only zeros before and after the data as the taper has not yet been
    # applied.
    assert tr.slice(endtime=length - 9).data.any()
    assert tr.slice(starttime=21).data.any()

    # Just taper everything
    taper_window(tr, 10, 20, taper_percentage=0.15, taper_type="cosine")

    # Only zeros before and after the data.
    assert not tr.slice(endtime=length - 9).data.any()
    assert not tr.slice(starttime=21).data.any()


def test_taper_window_arguments_passed_on():
    """
    Arguments should be passed to the underlying taper() function from obspy.
    """
    # Make sure any further arguments are passed to the taper function.
    tr = obspy.read()[0]
    tr.data = np.require(tr.data, dtype=np.float64)

    tr2 = tr.copy().taper(max_percentage=0.15, type="kaiser", beta=2.0)

    # Just taper everything
    taper_window(tr, 0, tr.stats.endtime - tr.stats.starttime,
                 taper_percentage=0.15, taper_type="kaiser", beta=2.0)

    # Deleting the processing information as this will not be the same.
    del tr.stats.processing
    del tr2.stats.processing
    assert tr == tr2
