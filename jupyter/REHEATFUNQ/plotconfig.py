# Configuration of the plots.
#
# This file is part of the REHEATFUNQ model.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
#               2022 Malte J. Ziebarth
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

from matplotlib import rcParams

# Font size 8 by default and use Roboto font:
rcParams['font.size'] = 8
rcParams['mathtext.it'] = 'Roboto:italic'
rcParams['mathtext.cal']= 'Roboto:italic'
rcParams["mathtext.fontset"] = 'custom'
rcParams["mathtext.default"] = 'it'
rcParams['mathtext.rm'] = 'Roboto'
rcParams["font.family"] = 'Roboto'

# Set backend to Retina for use on HiDPI screens:
try:
	from matplotlib_inline import backend_inline
	backend_inline.set_matplotlib_formats("retina")
except:
	pass
