===================================================
Pyadjoint
===================================================
`Pyadjoint <https://github.com/adjtomo/pyadjoint>`__ is an open-source Python
package for calculating time-dependent time series misfit, offering a number of
different misfit functions. It was designed to generate adjoint sources for 
full waveform inversion and adjoint tomography.

.. note::
    We recommend Pyadjoint within the larger misfit quantification package
    `Pyatoa <https://github.com/adjtomo/pyatoa>`__, as opposed to standalone.

Pyadjoint is hosted on `GitHub <https://github.com/adjtomo/pyadjoint>`__ as
part of the `adjTomo organization <https://github.com/adjtomo>`__.


Have a look at the `Pyadjoint usage <usage.html>`__ page to learn how
Pyadjoint is used

--------------

Installation
~~~~~~~~~~~~

It is recommended that Pyatoa be installed inside a `Conda 
<https://docs.conda.io/en/latest/>`__ environment.
The ``devel`` branch provides the latest codebase.

.. code:: bash

   git clone https://github.com/adjtomo/pyadjoint.git
   cd pyadjoint
   conda env create -n pyadjoint
   conda activate pyadjoint
   conda install obspy
   pip install -e .


Running Tests
`````````````

Tests ensure Pyadjoint runs as expected after changes are made to the source
code. You can run tests with Pytest.

.. code:: bash

   cd pyadjoint/tests
   pytest


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: INTRODUCTION

   usage
   example_dataset

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: MISFIT FUNCTIONS

   waveform
   convolution
   cctm
   mtm
   exponentiated_phase

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: DOUBLE DIFFERENCE MISFIT

   waveform_dd
   convolution_dd
   cctm_dd
   mtm_dd

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API