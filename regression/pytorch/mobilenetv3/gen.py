import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import geffnet

#torch_model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenetv3', pretrained=False)
#torch_model =  torch.load('./data/tf_mobilenetv3_small_075-da427f52.pth')
torch_model = geffnet.create_model('mobilenetv3_rw', pretrained=True)
torch_model.eval()
batch_size = 1    # just a random number
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "mobilenetv3_rw.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  verbose=True
                  )
print("Finish!")
