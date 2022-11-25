# Caching function calls according to hashed arguments.
#
# This file is part of the REHEATFUNQ model.
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

from pathlib import Path
from joblib import hash as jhash
from pickle import Pickler, Unpickler


def cached_call(fun, *args, **kwargs):
    """
    Executes a function call cached.
    """
    # Sort the keyword arguments:
    kwarg_hashes = [jhash((kw,kwargs[kw])) for kw
                    in sorted(kwargs.keys())]
    # Compute the argument hash:
    arghash = jhash((jhash(args), kwarg_hashes))


    # Ensure that the cache directory is created:
    Path(".cache").mkdir(exist_ok=True)

    # Filename of the cached result:
    fname = fun.__name__
    cache_dir = Path('.cache') / fun.__name__
    cached_file = cache_dir / (arghash + ".pickle")

    # Check if the file exists:
    if not cached_file.is_file():
        if not cache_dir.is_dir():
            cache_dir.mkdir()

        # Call the function:
        result = fun(*args, **kwargs)

        # Cache:
        with open(cached_file, 'wb') as f:
            Pickler(f).dump(result)

    else:
        with open(cached_file, 'rb') as f:
            result = Unpickler(f).load()

    return result
