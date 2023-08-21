# Test gamma conjugate prior code.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2023 Malte J. Ziebarth
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pytest
import numpy as np
from reheatfunq import GammaConjugatePrior
from reheatfunq.regional import default_prior

def test_gcp_kullback_leibler():
    """
    Test the Kullback-Leibler code.
    """
    # Make sure that using different amin raises errors:
    gcp0 = GammaConjugatePrior(3.0, 0.1, 0.1, 0.05, amin=1.0)
    gcp1 = GammaConjugatePrior(3.0, 0.1, 1.0, 0.05, amin=0.1)
    with pytest.raises(RuntimeError):
        gcp0.kullback_leibler(gcp1)
    with pytest.raises(RuntimeError):
        gcp0.kullback_leibler([gcp1])

    # Cases that should work out:
    gcp2 = GammaConjugatePrior(3.0, 0.1, 1.0, 0.05, amin=1.0)
    kl0 = gcp0.kullback_leibler(gcp2)
    kl1 = gcp0.kullback_leibler([gcp2])
    assert kl0 == kl1

    # Compare with a reference value to ensure portability and
    # consistency across versions:
    kl_ref = 87169608.06187229
    assert abs(kl0 - kl_ref) < 1e-6 * kl_ref
