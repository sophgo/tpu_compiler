from .base_model import model_base
import pymlir
import numpy as np



class MLIRModel(model_base):
    def __init__(self):
        self.net = pymlir.module()

    def load_model(self, mlirfile):
        self.net.load(mlirfile)

    def inference(self, input):
        assert(len(input) == 1)
        name = list(input.keys())[0]
        return self.net.run(input[name])

    def get_all_tensor(self, input_data=None):
        tensors_dict = self.net.get_all_tensor()
        return tensors_dict

    def get_op_info(self):
        return self.net.op_info