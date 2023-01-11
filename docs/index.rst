===================================================
Pyadjoint
===================================================
`Pyadjoint <https://github.com/adjtomo/pyadjoint>`__ is an open-source Python
package for calculating time-dependent misfit with a variety of misfit
functions. It was designed to generate adjoint sources for full waveform
inversion and adjoint tomography.

.. note::
    Although it can be used standalone, we recommend using Pyadjoint within
    the larger misfit quantification package
    `Pyatoa <https://github.com/adjtomo/pyatoa>`__.

Pyadjoint is hosted on `GitHub <https://github.com/adjtomo/pyadjoint>`__ as
part of the `adjTomo organization <https://github.com/adjtomo>`__.

Have a look at the `Pyadjoint usage <usage.html>`__ page to get started, and
browse available adjoint sources using the navigation bar.

--------------

Installation
~~~~~~~~~~~~

We recommend Pyadjoint be installed inside a `Conda
<https://docs.conda.io/en/latest/>`__ environment.

.. code:: bash

   git clone https://github.com/adjtomo/pyadjoint.git
   cd pyadjoint
   conda env create -n pyadjoint
   conda activate pyadjoint
   conda install obspy
   pip install -e .

--------------



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
   exp_phase

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: DOUBLE DIFFERENCE MISFIT

   dd_waveform
   dd_convolution
   dd_cctm
   dd_mtm

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: MISCELLANEOUS

   new_adjsrc
   citations
