"""
Test suite for Config class
"""
import inspect
import pytest
import pkgutil
import os

from pyadjoint import (get_config, get_function, adjoint_source_types,
                       ADJSRC_TYPES)


def test_all_configs():
    """Test importing all configs based on available types"""
    for adj_src_type in ADJSRC_TYPES.keys():
        get_config(adjsrc_type=adj_src_type, min_period=1, max_period=10)


def test_config_set_correctly():
    """Just make sure that choosing a specific adjoint source type exposes
    the correct parameters
    """
    cfg = get_config(adjsrc_type="multitaper", min_period=1,
                     max_period=10)
    # Unique parameter for MTM
    assert(hasattr(cfg, "phase_step"))


def test_incorrect_inputs():
    """
    Make sure that incorrect input parameters are error'd
    """
    with pytest.raises(AssertionError):
        get_config(adjsrc_type="cc_timetraveling_muskrat", min_period=1,
                   max_period=10)

    with pytest.raises(AssertionError):
        get_config(adjsrc_type="cc_traveltime_misfit", min_period=10,
                   max_period=1)


def test_adjsrc_types():
    """
    Check that all adjoint sources have the correct format
    """
    path = os.path.join(
        os.path.dirname(inspect.getfile(inspect.currentframe())),
        "adjoint_source_types")
    for importer, modname, _ in pkgutil.iter_modules(
            [path], prefix=adjoint_source_types.__name__ + "."):
        m = importer.find_module(modname).load_module(modname)
        name = m.__name__.split(".")[-1]

        assert hasattr(m, "calculate_adjoint_source"), \
            f"{name} has not function 'calculate_adjoint_source'"


def test_get_functions():
    """Get correct functions based on adjoint source types"""
    for adjsrc_type in ADJSRC_TYPES.keys():
        assert(callable(get_function(adjsrc_type)))
