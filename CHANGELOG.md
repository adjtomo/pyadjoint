Changelog
---------
Updates to Pyadjoint by version number

## v0.1.0
- Renamed 'pyadjoint/doc/' -> 'pyadjoint/docs'
- Renamed 'pyadjoint/src/' -> 'pydjoint/pyadjoint'
- Removed 'setup.py' in favor of new 'pyproject.toml'
- Updated setup URLs for dev team and website
- New 'main.py' script which now holds all the main processing scripts for
  discovering, calculating and plotting adjoint sources
- Reorganized main dir structure to include a utils/ directory which now 
  houses `dpss` and `utils`. Utils script split into multiple files for 
  increased organization.
- Renamed `dpss` utility functions to `mtm` for multitaper measurement
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
	  Now has a factor of 1/2 introduced so misfit is 1/2 original value.
	- Retains window-specific information and returns in the `ret_val` dict
	- Introduces a new `window_stats` boolean flag to toggle above return
- Cross-correlation Traveltime Misfit:
	- Retains window-specific information and returns in `ret_val` dict
	- Moves utility functions into `pyadjoint.utils` 
- Multitaper Misfit:
	- Overhauled function into a class to avoid the excessive parameter passing 
	  between functions, removed unncessary parameter renaming in-functino 
	- Moved 'frequency_limit' function into 'mtm' utils and cleaned up. Removed
	  unncessary functions which were passing too many variables, and included
	  doc string. Changed some internal parameter names to match the Config
	- Bugfix frequency limit search was not returning a `search` parameter flag
	  which is meant to stop the search (from Princeton version)
- Generic plot changed ylim of figure to max of waveform and not whole plot 
  because windows were plotted much larger than waveforms
