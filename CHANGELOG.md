Changelog
---------
Updates to Pyadjoint by version number

## v0.1.0
### Package structure
- Renamed 'pyadjoint/doc/' -> 'pyadjoint/docs'
- Renamed 'pyadjoint/src/' -> 'pydjoint/pyadjoint'
- Removed 'setup.py' in favor of new 'pyproject.toml'
- Updated setup URLs for dev team and website
- Reorganized main dir structure to include a utils/ directory which now 
  houses `dpss` and `utils`. Utils script split into multiple files for 
  increased organization.
- Revamped documentation, removed the old notebook and source-code bundled 
  documentation, moved all docs-related info out of source code and into the 
  upper-level docs/ directory.
- New 'furo' theme for documentation

### Bug fixes
- Plotting: changed `ylim` of figure to max of waveform and not whole plot 
  because windows were plotted much larger than waveforms

### New features
- Double difference capabilities added. 
  - These are placed directly into the respective adjoint sources rather than 
  	separately (as in Princenton/Mines version) to cut down on redundant code
  - Controlled with string flag ``choice``=='double_difference'. Requires 
    additional  waveforms and windows (``observed_dd``, ``synthetic_dd``, 
    ``windows_dd``)
  - Main caculate function will return two adjoint sources if 
    ``double_difference`` is set True
- Waveform misfit now includes ``convolved`` and ``double_difference`` 
  station pair referential adjoint sources
- New Misfit Functions:
	- Exponentiated phase
    - Convolved waveform difference
    - Waveform double difference
    - Convolved waveform double difference
	- Multitaper double difference
- Config class: 
	- Introduces individual Configs for each misfit function (as in Princeton v)
	- New `get_config` function which provides single entry point to grab 
	  Config objects by name
	- Config now carries the `adjsrc_type` parameter 

### Code architecture
- New ``main.py`` script which now holds all the main processing scripts for
  discovering, calculating and plotting adjoint sources
- Changed `window` -> `windows` for list of tuples carrying window information
- Moved plotting functions out of individual adjoint sources and into the
  main call function
- All adjoint sources return window information with a boolean toggle in 
  ``window_stats``. Information on window length, misfit, and type.
- All adjoint sources baselined with test data against Princeton/Mines version
  of code
- Removed plot capability for User to provide their own Figure object
- Removed boolean return flag for adjoint source and measurements because these
  are calculated regardless so might as well return them
- Config now carries information on `adjsrc_type`, removed this from the main
  calcualte_adjoint_source function, as it was redundnat. 

### Class/Function specific
- AdjointSource class:
	- New `window` attribute which holds information about input windows used 
	  to build the adjoint source, and their individual misfit values
	- Print statement now deals with `window attribute`
	- Write function cleaned up to include both SPECFEM and ASDF write functions
- Waveform Misfit adjoint source: 
	- Edited calculation of Simpson's rule integration to match Princeton ver.
	  Now has a factor of 1/2 introduced so misfit is 1/2 original value.
	- Baselined with test data against Priceton/Mines version of code
- Cross-correlation Traveltime Misfit:
	- Moves bulk of utility/calculation functions to ``pyadjoint.utils.cctm`` 
      so that MTM can use them allowing for less redundant code
- Multitaper Misfit:
	- Overhauled function into a class to avoid the excessive parameter passing 
	  between functions, removed unncessary parameter renaming in-function
	- Renamed many of the internal variables and functions for clarity
	- Removed redundant CCTM misfit definitions, instead switched to import of
	  functions from CCTM utility
	- Moved 'frequency_limit' function into 'mtm' utils and cleaned up. Removed
	  unncessary functions which were passing too many variables, and included
	  doc string. Changed some internal parameter names to match the Config
	- Bugfix frequency limit search was not returning a `search` parameter flag
	  which is meant to stop the search (from Princeton version)


