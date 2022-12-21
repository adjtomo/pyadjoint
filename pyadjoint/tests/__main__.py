#!/usr/bin/env python3
"""
The only purpose of this file is to be able to run the pyadoint test suite with

python -m pyadoint.tests

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import inspect
import os
import pytest
import sys


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))))

    sys.exit(pytest.main(PATH))
