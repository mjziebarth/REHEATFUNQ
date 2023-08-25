# Test R-disk code.
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
from reheatfunq.coverings.rdisks import all_restricted_samples

def test_all_restricted_samples():
    """

    """
    def compare_result(res, res_cmp):
        """
        This function compares the results to a benchmark result.
        """
        res = [(x[0], tuple(int(xi) for xi in x[1])) for x in res_cmp]
        res = list(sorted(res))

        # Equal number of elements:
        assert len(res) == len(res_cmp)

        # Equal probabilities and samples:
        assert all(x == y for x,y in zip(res, res_cmp))

    # X--X--X
    xy = np.array([(-2.0, 0.0), (0.0, 0.0), (2.0, 0.0)])
    res = all_restricted_samples(xy, 1.1, 100, 10000, 127, True)
    res_cmp = [(1.0, (0,1,2))]
    compare_result(res, res_cmp)

    rng = np.random.default_rng(893289)

    # X-X-X
    xy = np.array([(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)])
    res = all_restricted_samples(xy, 1.1, 100, 10000, rng, True)
    res_cmp = [(1/3, (1,)), (2/3, (0, 2))]
    compare_result(res, res_cmp)

    #   X
    #   |
    # X-X-X
    xy = np.array([(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    res = all_restricted_samples(xy, 1.1, 100, 10000, rng)
    res_cmp = [(1/4, (1,)), (3/4, (0, 2, 3))]
    compare_result(res, res_cmp)

    # X-XX-X
    xy = np.array([(-1.0, 0.0), (-0.05, 0.0), (0.05, 0.0), (1.0, 0.0)])
    res = all_restricted_samples(xy, 1.2, 100, 10000, rng, True)
    res_cmp = [(1/4, (1,)), (1/4, (2,)), (1/2, (0, 3))]
    compare_result(res, res_cmp)

    #   X
    #  / \
    # X - X  X-X  X
    xy = np.array([(-0.5, 0.0), (0.5, 0.0), (0.0, np.sqrt(3)/2),
                   (2.0, 0.0), (3.0, 0.0), (5.0, 0.0)])
    res = all_restricted_samples(xy, 1.2, 100, 10000, rng, True)
    res_cmp = [(1/6, (0,3,5)), (1/6, (0,4,5)), (1/6, (1,3,5)), (1/6, (1,4,5)),
               (1/6, (2,3,5)), (1/6, (2,4,5)),]
    compare_result(res, res_cmp)

    # A set of 40 randomly distributed points
    # Almost certainly raises an exception.
    xy = np.random.random((40,2))
    res = all_restricted_samples(xy, 0.1, 10000, 100000, rng, True)
    with pytest.raises(RuntimeError):
        res = all_restricted_samples(xy, 0.1, 100, 100, None, True)

    # A set of 100 randomly distributed points
    xy = np.random.random((100,2))
    res = all_restricted_samples(xy, 0.1, 10000, 100000, rng, True)
