# Resnet18

# Convert to Onnx
this model built-in torch api, we don't need to download by self

```
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

torch_model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
torch_model.eval()
batch_size = 1    # just a random number
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
print("Finish!")

```

get a onnx model

