print("")
print("==== numpy include ====")
import sys
print("path:",sys.path)
import numpy as np
import subprocess
subprocess.run(['whereis', 'python'])
incl = np.get_include()
print(incl)

print("==== numpy include ====")
print("")
