from functools import partial, wraps
import torch

# generate onnx model from torch
def to_onnx(torch_model, input, model_path, inputs_list, outputs_list):
    torch.onnx.export(torch_model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = inputs_list,   # the model's input names
                  output_names = outputs_list, # the model's output names
                  )
class Color:
    # Foreground:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    # Formatting
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # End colored text
    END = '\033[0m'
    NC = '\x1b[0m'  # No Color



def docolor(color):
    def decorator(func):
        def wrap(*args, **kwargs):
            if color == 'blue':
                print(Color.OKBLUE)
            elif color == 'green':
                print(Color.OKGREEN)
            elif color == 'yellow':
                print(Color.WARNING)
            elif color == 'red':
                print(Color.FAIL)
            f = func(*args)
            print(Color.END)
            return f
        return wrap
    return decorator

