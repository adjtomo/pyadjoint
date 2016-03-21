#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
import logging


class PyadjointError(Exception):
    """
    Base class for all Pyadjoint exceptions. Will probably be used for all
    exceptions to not overcomplicate things as the whole package is pretty
    small.
    """
    pass


class PyadjointWarning(UserWarning):
    """
    Base class for all Pyadjoint warnings.
    """
    pass


__version__ = "0.0.1a"


# setup the logger
logger = logging.getLogger("pyadjoint")
logger.setLevel(logging.WARNING)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Main objects and functions available at the top level.
from .adjoint_source import AdjointSource, calculate_adjoint_source, calculate_adjoint_source_DD  # NOQA #YY
from .config import Config  # NOQA
