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
        ['python','-c','import numpy; print(numpy.get_include())'],
        env={}
    )

    np_include = np_incl_byte.decode().strip()

    # Make numpy available in this isolated path:
    success = False
    print(sys.path[::-1])
    for path in sys.path[::-1]:
        if 'site-packages' in path:
            p = Path(path)
            is_dir = (p / "numpy").is_dir()
            if is_dir:
                rename((p / "numpy").resolve(), (p / "numpyold").resolve())
            print(subprocess.check_output(['ls','-la',str(p.resolve())]).decode())
            symlink(Path(np_include).parent.parent.resolve(),
                    (p / "numpy").resolve())
            success = True
            break

    if not success:
        raise RuntimeError("Could not link the NumPy package to the isolated "
                           "site-packages.")

    print(np_include)
