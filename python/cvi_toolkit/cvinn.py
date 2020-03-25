import numpy as np
import onnx
from .model import CVI_Model
from .transform import OnnxConverter
import subprocess


class cvinn(object):
    def __init__(self):
        pass
    def load_model(self, model_type: str, model_file: str,  mlirfile: str, weight_file: str = None):
        if model_type == 'caffe':
            if weight_file == None:
                print("No caffe weight file")
                return -1
            subprocess.run(["mlir-translate", "--caffe-to-mlir", model_file,
                    "--caffemodel", weight_file,
                    "-o", mlirfile
                    ])
            return 0
        elif model_type == 'onnx':
            if not model_file.lower().endswith('.onnx'):
                print("{} is not end with .onnx".format(model_file))
                return -1

            onnx_model = onnx.load(model_file)
            c = OnnxConverter(model_file.split('.')[0].split('/')[-1], onnx_model, mlirfile)
            c.run()
            return 0
        else:
            print("Not support {} type, now support onnx and caffe".format(model_type))
            return -1

    def calibration(self):
        return 0

    def build_cvimodel(self):
        return 0

    def tpu_simulation(self):
        return 0

    def inference(self, model_type: str, input_data: np.ndarray, model_file=None, weight_file=None, mlirfile=None, all_tensors:str = None):
        net = CVI_Model()
        print(model_type)
        net.load_model(model_type, model_file=model_file, weight_file=weight_file, mlirfile=mlirfile)
        out = net.inference(input_data)
        if all_tensors!=None:
            net.get_all_tensor(all_tensors)
        return out
    def cleanup(self):
        return 0
    def dump_quant_info(self):
        return 0
