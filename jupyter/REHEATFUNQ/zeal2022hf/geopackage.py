# Read from a geopackage.
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

import numpy as np
import geopandas as gpd
from pyproj import Proj

def read_geopackage_polys(fname, gpkg_projstr):
    # Read the data:
    data = gpd.read_file(fname)

    # Get the Polygons:
    polys = [d[2].exterior.coords.xy for d in data.to_numpy()]

    # To geographic coordinates:
    proj = Proj(gpkg_projstr)
    polys = [np.array(proj(*p, inverse=True)).T for p in polys]

    return polys
