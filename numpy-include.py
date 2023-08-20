#
# Invoke Python with empty environement variables to break out of the
# build isolation.
#
try:
    import numpy
    print(numpy.get_include())

except ImportError:
    import subprocess
    np_incl_byte = subprocess.check_output(
        ['python','-c','import numpy; print(numpy.get_include())'],
        env={}
    )

    print(np_incl_byte.decode().strip())
