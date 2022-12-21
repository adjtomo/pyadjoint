"""
Test suite for Config class
"""
import pytest
from pyadjoint import config
from pyadjoint.utils import discover_adjoint_sources


def test_all_configs():
    """Test importing all configs based on available types"""
    adj_src_types = discover_adjoint_sources().keys()
    for adj_src_type in adj_src_types:
        config(adjsrc_type=adj_src_type, min_period=1, max_period=10)


def test_config_set_correctly():
    """Just make sure that choosing a specific adjoint source type exposes
    the correct parameters
    """
    cfg = config(adjsrc_type="multitaper_misfit", min_period=1, max_period=10)
    # Unique parameter for MTM
    assert(hasattr(cfg, "phase_step"))


def test_incorrect_inputs():
    """
    Make sure that incorrect input parameters are error'd
    """
    with pytest.raises(NotImplementedError):
        config(adjsrc_type="cc_timetraveling_muskrat", min_period=1,
               max_period=10)

    with pytest.raises(AssertionError):
        config(adjsrc_type="cc_traveltime_misfit", min_period=10, max_period=1)
