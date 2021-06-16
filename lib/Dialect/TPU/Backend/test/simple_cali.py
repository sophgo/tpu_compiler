import numpy as np
import sys

npz_file = sys.argv[1]

tensors = np.load(npz_file)
for t in tensors:
    threshold = np.abs(np.max(tensors[t]))+0.0001
    if threshold >= 1000000.0:
        threshold = 1000000.0
    elif np.isnan(threshold) or threshold == 0.0:
        threhsold = 0.1
    else:
        pass
    print("{} {}".format(t,threshold))
