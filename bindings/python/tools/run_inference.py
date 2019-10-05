#!/usr/bin/python

import sys
import numpy as np
import pymlir

module = pymlir.module()
module.load(sys.argv[1])
print("load module done")
# module.dump()

