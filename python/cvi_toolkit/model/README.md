# CVI Model

use python interface to inference cvi mlir model

example:

```
from cvi_toolkit.model import CVI_MODEL

net = CVI_MODEL()
net.NN_RegisterModel("efficientnet_b0_opt.mlir")
net.NN_Forward(input)
net.get_all_tensor()
```

or

```
from cvi_toolkit.model import CVI_MODEL

net = CVI_MODEL(register_model="efficientnet_b0_opt.mlir")
net.NN_Forward(input)
net.get_all_tensor()
```