Changelog
---------
Updates to Pyadjoint by version number

## v0.1.0
- Renamed 'pyadjoint/doc/' -> 'pyadjoint/docs'
- Renamed 'pyadjoint/src/' -> 'pydjoint/pyadjoint'
- Removed 'setup.py' in favor of new 'pyproject.toml'
- Updated setup URLs for dev team and website
- New 'main.py' script which now holds all the main processing scripts (e.g., 
  'calculate_adjoint_source.py')
- AdjointSource:
	- New `window` attribute which holds information about input windows used 
	  to build the adjoint source, and their individual misfit values
	- Print statement now deals with `window attribute`
	- Write function cleaned up to include both SPECFEM and ASDF write functions
- Config: 
	- Introduces individual Configs for each misfit function (as in Princeton v)
	- New `get_config` function which provides single entry point to grab 
	  Config objects by name
- Waveform Misfit: 
	- Edited calculation of Simpson's rule integration to match Princeton ver.
	- Retains window-specific information and returns in the `ret_val` dict
	- Introduces a new `window_stats` boolean flag to toggle above return
- Cross-correlation Traveltime Misfit:
	- 
