"""
Test generalized adjoint source generation for each type
"""
import pytest
import numpy as np
from pyadjoint import calculate_adjoint_source, get_config
from pyadjoint import get_example_data

path = "/Users/chow/Work/pyadjoint/new_images"  # deleteme


@pytest.fixture
def example_data():
    """Return example data to be used to test adjoint sources"""
    obs, syn = get_example_data()
    obs = obs.select(component="Z")[0]
    syn = syn.select(component="Z")[0]

    return obs, syn


@pytest.fixture
def example_2_data():
    """
    Return example data to be used to test adjoint source double difference
    calculations. Simply grabs the R component to provide a different waveform
    """
    obs, syn = get_example_data()
    obs = obs.select(component="R")[0]
    syn = syn.select(component="R")[0]

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
        windows=example_window, plot=True,
        plot_filename=f"{path}/waveform_misfit.png"
    )
    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0
    assert len(adjsrc.windows) == 1
    assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_waveform_dd_misfit(example_data, example_2_data, example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    obs_2, syn_2 = example_2_data
    cfg = get_config(adjsrc_type="waveform_misfit", min_period=30.,
                     max_period=75.)
    adjsrcs = calculate_adjoint_source(
        adj_src_type="waveform_misfit", observed=obs, synthetic=syn, config=cfg,
        windows=example_window,  plot=True,
        plot_filename=f"{path}/waveform_2_misfit.png",
        choice="double_difference", observed_2=obs_2, synthetic_2=syn_2,
        windows_2=example_window
    )
    assert(len(adjsrcs) == 2)
    for adjsrc in adjsrcs:
        assert adjsrc.adjoint_source.any()
        assert adjsrc.misfit >= 0.0
        assert len(adjsrc.windows) == 1
        assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_convolved_waveform_misfit(example_data, example_2_data,
                                   example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    obs_2, syn_2 = example_2_data
    cfg = get_config(adjsrc_type="waveform_misfit", min_period=30.,
                     max_period=75.)
    adjsrc = calculate_adjoint_source(
        adj_src_type="waveform_misfit", observed=obs, synthetic=syn, config=cfg,
        windows=example_window,  plot=True,
        plot_filename=f"{path}/waveform_2_misfit.png",
        choice="convolved", observed_2=obs_2, synthetic_2=syn_2,
        windows_2=example_window
    )
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
        config=cfg, windows=example_window, plot=True,
        plot_filename=f"{path}/cc_traveltime_misfit.png",
        choice="double_difference", 
    )

    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0
    assert len(adjsrc.windows) == 1
    assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_dd_cc_traveltime_misfit(example_data, example_2_data, example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    obs_2, syn_2 = example_2_data
    cfg = get_config(adjsrc_type="cc_traveltime_misfit", min_period=30.,
                     max_period=75.)
    adjsrcs = calculate_adjoint_source(
        adj_src_type="cc_traveltime_misfit", observed=obs, synthetic=syn,
        config=cfg, windows=example_window, plot=True,
        plot_filename=f"{path}/cc_traveltime_misfit.png",
        choice="double_difference", observed_2=obs_2, synthetic_2=syn_2,
        windows_2=example_window
    )

    assert(len(adjsrcs) == 2)
    for adjsrc in adjsrcs:
        assert adjsrc.adjoint_source.any()
        assert adjsrc.misfit >= 0.0
        assert len(adjsrc.windows) == 1
        assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_multitaper_misfit(example_data, example_window):
    """
    Test the waveform misfit function
    """
    from pyadjoint import logger
    logger.setLevel("DEBUG")

    obs, syn = example_data
    cfg = get_config(adjsrc_type="multitaper_misfit", min_period=30.,
                     max_period=75., min_cycle_in_window=3., 
                     use_cc_error=False)

    adjsrc = calculate_adjoint_source(
        adj_src_type="multitaper_misfit", observed=obs, synthetic=syn,
        config=cfg, windows=example_window, plot=False,
        plot_filename=f"{path}/multitaper_misfit.png"
    )

    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0
    # Make sure the adj src successfully uses MTM measurement, does not fall
    # back to cross correlation traveltime
    for stats in adjsrc.window_stats:
        assert(stats["type"] == "multitaper")

    assert isinstance(adjsrc.adjoint_source, np.ndarray)


def test_exponentiated_phase_misfit(example_data, example_window):
    """
    Test the waveform misfit function
    """
    obs, syn = example_data
    cfg = get_config(adjsrc_type="exponentiated_phase_misfit", min_period=30.,
                     max_period=75.)

    adjsrc = calculate_adjoint_source(
        adj_src_type="exponentiated_phase_misfit", observed=obs, synthetic=syn,
        config=cfg, windows=example_window, plot=True,
        plot_filename=f"{path}/exponentiated_phase_misfit.png"
    )

    assert adjsrc.adjoint_source.any()
    assert adjsrc.misfit >= 0.0

    assert isinstance(adjsrc.adjoint_source, np.ndarray)

