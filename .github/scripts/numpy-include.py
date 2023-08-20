print("")
print("==== numpy include ====")
import sys
print("path:",sys.path)
import numpy as np
import subprocess
print(subprocess.check_output(['whereis', 'python']))
incl = np.get_include()
print(incl)

print("==== numpy include ====")
print("")
