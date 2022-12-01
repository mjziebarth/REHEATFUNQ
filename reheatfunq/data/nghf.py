# A function to read the NGHF data base.
#
# This file is part of the ziebarth_et_al_2022_heatflow python module.
#
# Author: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2022 Deutsches GeoForschungsZentrum Potsdam,
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

from math import inf
from csv import reader

def read_nghf(f):
    """
    This function reads the NGHF data base.

    Parameters
    ----------
    f : str | pathlib.Path
        The file path to the ``NGHF.csv`` file of Lucazeau [L2019]_.

    Returns
    -------
    nghf_lon : list
       Longitudes of the data points. This field is mandatory.
    nghf_lat : list
       Latitudes of the data points. This field is mandatory.
    nghf_hf : list
       Heat flow of the data points in :math:`\mathrm{mWm}^{-2}`.
       This field is mandatory.
    nghf_quality : list
       Quality of the data points. If defined, one of 'A', 'B', 'C',
       'D' or 'Z'.
    nghf_yr : list
       Measurement year of the data points. This field is
       mandatory.
    nghf_type : list
       Whether the heat flow data point is continental or oceanic.
       If defined, one of 'land' or 'ocean'. This field is
       mandatory.
    nghf_max_depth : list
       Maximum depth used in estimating the temperature gradient.
       If the field is empty, returns -9999999.
    nghf_uncertainty : list
       Relative uncertainty of the data points. Returns 0.1 for
       'A' quality, 0.2 for 'B' quality, 0.3 for 'C' quality,
       and infinity otherwise.
    indexmap : dict
       Maps indices within the returned arrays to lines in the
       ``NGHF.csv`` table. If :code:`i` was an index within the
       returned arrays, then :code:`j = indexmap[i]` is the row
       within the NGHF table.
    """
    # Create the data containers:
    nghf_names = []
    nghf_quality = []
    nghf_lon = []
    nghf_lat = []
    nghf_hf = []
    nghf_yr = []
    nghf_type = []
    nghf_max_depth = []
    nghf_uncertainty = []
    indexmap = {}
    failures = 0
    nghf_mask_readable = []

    # Read the table:
    with open(f,'r') as csvfile:
        table = reader(csvfile, strict=True)

        # Parse the table:
        for i,row in enumerate(table):
            # The required data:
            try:
                hf = float(row[19])
                lon = float(row[8])
                lat = float(row[9])
                yr = int(row[24])
            except:
                failures += 1
                nghf_mask_readable.append(False)
                continue

            # Maximum depth with a drop-in for missing data:
            try:
                max_depth = int(row[12])
            except:
                # Set an obvious error flag here, still retaining the
                # data for further analysis:
                max_depth = -9999999

            # Uncertainty with drop-in:
            try:
                uncertainty = float(row[20])
            except:
                uncertainty = inf

            # Type:
            # Code 1 of the table is a geographic information, dividing the data
            # set into different continental / oceanic areas.
            # See the supporting information S1, table S1 for information:
            if row[1] in ('A','B','C','D','E','F','G','H'):
                tp = 'land'
            elif row[1] in ('N','O','P','Q','R','S','X','Y'):
                tp = 'ocean'
            else:
                failures += 1
                nghf_mask_readable.append(False)
                continue

            # Valid data point!
            indexmap[len(nghf_lon)] = i
            nghf_mask_readable.append(True)
            nghf_names.append(row[7])
            nghf_lon.append(lon)
            nghf_lat.append(lat)
            nghf_quality.append(row[6])
            nghf_hf.append(hf)
            nghf_yr.append(yr)
            nghf_type.append(tp)
            nghf_max_depth.append(max_depth)
            nghf_uncertainty.append(uncertainty)

    # Translate the quality encoding to variance:
    nghf_quality_var   = [{'A' : 0.1, 'B' : 0.2, 'C' : 0.3, 'D' : inf,
                           'Z' : inf, '' : inf}[q] for q in nghf_quality]

    return nghf_lon, nghf_lat, nghf_hf, nghf_quality, nghf_yr, nghf_type, \
           nghf_max_depth, nghf_uncertainty, indexmap


