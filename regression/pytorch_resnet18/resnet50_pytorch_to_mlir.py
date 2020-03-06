import torch
import torch.onnx
from collections import namedtuple
from PIL import Image
from torchvision import transforms
import inspect
import numpy as np 
import logging
from onnx import onnx, numpy_helper
from transform.onnx_converter import OnnxConverter

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-4s %(filename)2s %(lineno)d: %(message)s',
                    datefmt='%m-%d %H:%M')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
input_image = Image.open('data/dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
# read resnet 18
torch_model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=False)
torch_model.eval()

#inference pytorch
torch_out = torch_model(input_batch)

# export to onnx
torch.onnx.export(torch_model,               # model being run
                  input_batch,                         # model input (or a tuple for multiple inputs)
                  "sam.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['data'],   # the model's input names
                  output_names = ['output'], # the model's output names
)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# store pytorch golden
np.savez("input.npz", **{'data': to_numpy(input_batch)})
np.savez("golden_output.npz", **{'output': to_numpy(torch_out)})

# convert onnx to mlir 
onnx_model = onnx.load("sam.onnx")
c = OnnxConverter("ResNet-50-model", onnx_model)
c.run()

