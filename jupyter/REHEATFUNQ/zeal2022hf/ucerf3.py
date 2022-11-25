# Result of the UCERF3 branch analysis.
#
# This file is part of the ziebarth_et_al_2022_heatflow python module.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam
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

class UCERF3BranchResult:
    """
    This class contains the result of the analysis of one
    UCERF3 branch.
    """
    def __init__(self, weight, saf_relative_power, saf_power, total_power,
                 fault_power, ld2v, d2, sr):
        self.weight = weight
        self.saf_relative_power = saf_relative_power
        self.saf_power = saf_power
        self.total_power = total_power
        self.fault_power = fault_power
        self.ld2v = ld2v
        self.d2 = d2
        self.sr = sr
