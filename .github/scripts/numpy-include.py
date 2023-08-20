from time import sleep
print("")
print("==== numpy include ====")
import sys
print("path:",sys.path)
import numpy as np
import subprocess
print(subprocess.check_output(
    ['python','-c','import sys; print(sys.path)'],
    env={}))
print(subprocess.check_output(
    ['python','-c','import os; import numpy; print(numpy.get_include())'],
    env={}))
sleep(1)

incl = np.get_include()
print(incl)

print("==== numpy include ====")
print("")
