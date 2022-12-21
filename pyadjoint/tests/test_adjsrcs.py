"""
Test generalized adjoint source generation for each type
"""
import pytest
from pyadjoint import calculate_adjoint_source
from pyadjoint.utils import get_example_data


@pytest.fixture
def example_data():
    """Return example data to be used to test adjoint sources"""
    obs, syn = get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    return obs, syn


def test_calculate_waveform_adjoint_source():
    """

    """
    calculate_adjoint_source(adj_src_type="waveform_misfit")
