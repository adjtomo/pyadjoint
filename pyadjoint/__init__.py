#!/usr/bin/env python3
"""
:copyright:
    adjTomo Dev Team (adjtomo@gmail.com), 2022
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
import inspect
import logging
import os
import pkgutil

__version__ = "0.1.0"


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


def discover_adjoint_sources():
    """
    Discovers the available adjoint sources in the package. This should work
    regardless of whether Pyadjoint is checked out from git, packaged as .egg
    etc.
    """
    from pyadjoint import adjoint_source_types

    adjoint_sources = {}

    fct_name = "calculate_adjoint_source"

    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        if not hasattr(m, fct_name):
            continue
        fct = getattr(m, fct_name)
        if not callable(fct):
            continue

        # Create a dictionary of functions related to the adjsrc name
        name = m.__name__.split(".")[-1]
        adjoint_sources[name] = fct

    return adjoint_sources


# setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# Main objects and functions available at the top level.
from .adjoint_source import AdjointSource # NOQA
from .main import (calculate_adjoint_source, get_example_data,
                   discover_adjoint_sources, plot_adjoint_source)  # NOQA
from .config import get_config  # NOQA
