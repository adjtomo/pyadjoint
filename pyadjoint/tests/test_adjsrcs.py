"""
Test generalized adjoint source generation for each type
"""
import pytest
import numpy as np
from pyadjoint import calculate_adjoint_source, get_config
from pyadjoint.utils import get_example_data


@pytest.fixture
def example_data():
    """Return example data to be used to test adjoint sources"""
    obs, syn = get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    return obs, syn


@pytest.fixture
def example_window():
    """Defines an example window where misfit can be quantified"""
    return [[2076., 2418.0]]


def test_waveform_misfit(example_data, example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    cfg = get_config(adjsrc_type="waveform_misfit", min_period=30.,
                     max_period=75.)
    adjsrc = calculate_adjoint_source(
        adj_src_type="waveform_misfit", observed=obs, synthetic=syn, config=cfg,
        window=example_window, adjoint_src=True, plot=True,
        plot_filename="waveform_misfit.png"
    )
    pytest.set_trace()
    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0
    assert len(adjsrc.windows) == 1

    assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_cc_traveltime_misfit(example_data, example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    cfg = get_config(adjsrc_type="cc_traveltime_misfit", min_period=30.,
                     max_period=75.)
    adjsrc = calculate_adjoint_source(
        adj_src_type="cc_traveltime_misfit", observed=obs, synthetic=syn,
        config=cfg, window=example_window, adjoint_src=True, plot=True,
        plot_filename="cc_traveltime_misfit.png"
    )

    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0

    assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_multitaper_misfit(example_data, example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    cfg = get_config(adjsrc_type="multitaper_misfit", min_period=30.,
                     max_period=75.)
    adjsrc = calculate_adjoint_source(
        adj_src_type="multitaper_misfit", observed=obs, synthetic=syn,
        config=cfg, window=example_window, adjoint_src=True, plot=True,
        plot_filename="multitaper_misfit.png"
    )

    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0

    assert isinstance(adjsrc.adjoint_source, np.ndarray)

