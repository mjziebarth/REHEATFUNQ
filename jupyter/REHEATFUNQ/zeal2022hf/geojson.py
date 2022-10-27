# Code to write research results to GeoJSON format.
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

def write_polygon_geojson(fpath, lon, lat, data, fields='data', name='A GeoJSON file'):
    """
    Writes a GeoJSON file.
    Parameters:
      lon  : N-element list of longitudes
      lat  : N-element list of of latitutdes
      data : list of m-dimensional data.
    """
    geojson = '{\n' \
              '  "type": "FeatureCollection",\n' \
             f'  "name": "{name}",\n' \
              '  "crs": {\n' \
              '     "type": "name",\n' \
              '     "properties": {\n' \
              '        "name": "urn:ogc:def:crs:OGC:1.3:CRS84"\n' \
              '     }\n' \
              '  },\n' \
              '  "features": [\n'

    # Convert everything to lists:
    if not isinstance(lon,list):
        lon = list(lon)
    if not isinstance(lat,list):
        lat = list(lat)
    if not isinstance(data,list):
        data = list(data)

    # Sanity:
    N = len(lon)
    assert len(lat) == N

    # Number of fields:
    if isinstance(fields,str):
        fields = [fields]
    elif not isinstance(fields, list):
        fields = list(fields)
    m = len(fields)
    assert len(data) == m

    def property_formatter(p):
        if isinstance(p,str):
            return '"' + p + '"'
        if isinf(p):
            return '"inf"'
        return p

    # Feature header 
    feat  =  '    {\n' \
             '      "type": "Feature",\n' \
             '      "geometry": {\n' \
             '        "type": "Polygon",\n' \
             '        "coordinates": [[\n'
    # Iterate over the points and add as features:
    for i in range(N-1):
        # Feature coordinates:
        feat += f'         [{lon[i]},\n' \
                f'          {lat[i]}],\n'
    feat += f'         [{lon[N-1]},\n' \
            f'          {lat[N-1]}]\n' \
             '        ]]\n' \
             '      },\n'
    # All feature properties:
    feat += '      "properties": {\n'
    if m > 0:
        for j in range(m-1):
              feat += f'        "{fields[j]}": {property_formatter(data[i][j])},\n'
        feat += f'        "{fields[m-1]}": {property_formatter(data[i][m-1])}\n'
    feat += '      }\n'
    # Closing the feature:
    feat += '    }\n'
    
    geojson += feat
    geojson += '  ]\n' \
               '}'

    with open(fpath, 'w') as f:
        f.write(geojson)
