import numpy as np
import sys

npz_file = sys.argv[1]

tensors = np.load(npz_file)
for t in tensors:
    threshold = np.abs(np.max(tensors[t]))
    if np.isnan(threshold):
        threshold = 10.0
    elif threshold >= 127.0:
        threshold = 127.0
    elif threshold <= 0.001:
        threshold = 1.0
    else:
        pass
    print("{} {}".format(t,threshold))
