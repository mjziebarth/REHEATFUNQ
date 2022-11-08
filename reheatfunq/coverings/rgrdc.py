# Random global R-disk coverings.
#
# This file is part of the REHEATFUNQ model.
#
# Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
#
# Copyright (C) 2019-2022 Deutsches GeoForschungsZentrum Potsdam,
#                    2022 Malte J. Ziebarth
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
from typing import Optional
from numpy.typing import ArrayLike
from pyproj import Proj, Geod
from scipy.spatial import KDTree
from loaducerf3 import PolygonSelector, Polygon
from .rdisks import conforming_data_selection


def random_global_R_disk_coverings(R: float, min_points: int, hf: ArrayLike,
                                   buffered_poly_xy: list[ArrayLike],
                                   proj_str: str,  N: int = 10000,
                                   MAX_DRAW: int = 100000, dmin: float = 0.0,
                                   seed: int = 982981,
                                   used_points: Optional[list[int]] = None,
                                   a: float = 6378137.0):
    """
    Uses rejection sampling to draw a number of exclusive regional
    distributions.
    """
    rng = np.random.default_rng(seed)

    distributions = []
    distribution_indices = []
    n_draw = 0
    valid_points = []
    lolas = []

    if used_points is None:
        used_points = set()
    else:
        used_points = set(used_points)

    print("Area of a disk:          ", round(np.pi*R**2 * 1e-6), "km²")

    # The polygon selector:
    buffered_poly = Polygon(*buffered_poly_xy)
    selector = PolygonSelector(buffered_poly)
    xy_study_area = np.zeros((1,2))

    # Use efficient indexing in Euclidean space to get the neighbors:
    hf_lambda = np.deg2rad(hf[:,1])
    hf_phi = np.deg2rad(hf[:,2])
    xyz = np.stack((np.cos(hf_lambda) * np.cos(hf_phi),
                    np.sin(hf_lambda) * np.cos(hf_phi),
                    np.sin(hf_phi)), axis=1)
    tree = KDTree(xyz)
    # Maximum Euclidean distance
    # (2*asin(R/(2*a)) is exact on a sphere. Allow some margin to account
    # for the ellipsoid.)
    distmax = 1.3 * (2 * np.arcsin(0.5 * R/a))
    print("Number of heat flow data:", hf.shape[0])

    # Projection:
    proj_study_area = Proj(proj_str)

    # For the cleanup:
    geod = Geod(ellps='WGS84')

    # Progress logging:
    last_prog_5perc_N = -1
    last_prog_5perc_MAX_DRAW = -1

    for i in range(N):
        m = 0
        while m < min_points:
            # Log progress:
            log_msg = False
            if i >= (last_prog_5perc_N+1) / 20. * N:
                last_prog_5perc_N += 1
                log_msg = True
            if n_draw >= (last_prog_5perc_MAX_DRAW+1) / 20. * MAX_DRAW:
                last_prog_5perc_MAX_DRAW += 1
                log_msg = True
            if log_msg:
                print(" " + str(5 * last_prog_5perc_N) + "% distributions at "
                      + str(5 * last_prog_5perc_MAX_DRAW) + "% of maximum "
                      "disk draws.")


            # Exit condition:
            if n_draw > MAX_DRAW:
                break

            # Draw a random point on the sphere:
            point = (360.*rng.random()-180.,
                     np.rad2deg(np.arcsin(2*rng.random()-1)))
            n_draw += 1

            # Test whether it intersects the study area:
            xy_study_area[0,:] = proj_study_area(*point)
            if selector.array_mask(xy_study_area)[0]:
                continue

            # Compute the Euclidean coordinates:
            p_lambda, p_phi = np.deg2rad(point)
            cos_phi = np.cos(p_phi)
            point_xyz = (np.cos(p_lambda) * cos_phi,
                         np.sin(p_lambda) * cos_phi,
                         np.sin(p_phi))

            # Now query all within distance of that node:
            neighbors = tree.query_ball_point(point_xyz, distmax)
            if len(neighbors) < min_points:
                continue

            lon_cmp = np.ones(len(neighbors))
            lat_cmp = np.ones(len(neighbors))
            lon_cmp *= point[0]
            lat_cmp *= point[1]
            distances = geod.inv(lon_cmp, lat_cmp, hf[neighbors,1],
                                 hf[neighbors,2])[2]
            neighbors = np.array(neighbors)[distances <= R]

            # Make sure that we do not count nodes twice:
            neighbors = neighbors[[n not in used_points for n in neighbors]]

            # If we exceed the required size, obtain heat flow data set and its
            # coordinates:
            m = neighbors.size
            if m >= min_points:
                hfn = hf[neighbors,:]

                # Apply the minimum distance filtering:
                proj = Proj(f'+proj=stere +lon_0={point[0]} +lat_0={point[1]}'
                             ' +ellps=WGS84')
                xyn = np.array(proj(*hfn[:,1:3].T)).T.copy()
                mask = conforming_data_selection(xyn, dmin, rng=rng)
                # Apply the mask if needed:
                m = np.count_nonzero(mask)
                if m != neighbors.size:
                    neighbors = neighbors[mask]
                    lola = hfn[mask,1:3]
                    hfn = hfn[mask,0]


        if n_draw > MAX_DRAW:
            break

        # Remember the data points that we have already drawn:
        neighbors = list(neighbors)
        used_points.update(neighbors)

        # Add the point to the valid points:
        valid_points.append(point)

        # Obtain the heat flow distribution:
        distributions.append(np.sort(hfn).reshape(-1))
        distribution_indices.append(neighbors)
        lolas.append(lola)

    print("Disk rejection rate:     ", 100.*(n_draw-N) / n_draw)

    return valid_points, used_points, distributions, lolas, distribution_indices
