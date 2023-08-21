#
# Invoke Python with empty environement variables to break out of the
# build isolation.
#
try:
    import numpy
    print(numpy.get_include())

except ImportError:
    import subprocess
    import sys
    from pathlib import Path
    from os import rename, symlink
    np_incl_byte = subprocess.check_output(
        [sys.executable,'-c','import numpy; print(numpy.get_include())'],
        env={}
    )

    np_include = np_incl_byte.decode().strip()

    # Make numpy available in the isolated environment this code is running
    # in. Do this by symlinking the previously discovered system NumPy
    # package into a site-package directory found in the isolated environment
    # Python path.
    found_site_packages = False
    success = False
    for path in sys.path[::-1]:
        p = Path(path)
        if 'site-packages' in path:
            found_site_packages = True
            p = Path(path)
            if not p.exists():
                continue
            is_dir = (p / "numpy").is_dir()
            if is_dir:
                rename((p / "numpy").resolve(), (p / "numpyold").resolve())
            symlink(Path(np_include).parent.parent.resolve(),
                    (p / "numpy").resolve())
            success = True
            break

    if not success:
        msg = "Could not link the NumPy package to the isolated site-packages."
        if found_site_packages:
            msg += " Found site-packages but did not exist."
        else:
            msg += " Found no site-pacakges."
        raise RuntimeError(msg)

    print(np_include)
