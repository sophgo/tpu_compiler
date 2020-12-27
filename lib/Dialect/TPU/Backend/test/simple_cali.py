import numpy as np
import os
import sys

npz_file = sys.argv[1]

tensors = np.load(npz_file)
for t in tensors:
    print("{} {}".format(t,np.max(tensors[t])+0.0001))
