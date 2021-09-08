from .base_model import model_base
import pymlir
import numpy as np



class MLIRModel(model_base):
    def __init__(self):
        self.net = pymlir.module()

    def load_model(self, mlirfile):
        self.net.load(mlirfile)

    def inference(self, inputs):
        if isinstance(inputs, dict):
            for k,v in inputs.items():
                self.net.set_tensor(k, v)
            self.net.invoke()
            all_tensor = self.net.get_all_tensor()
            output_dit = {}
            for key, value in all_tensor.items():
                if key == self.net.get_output_details()[0]:
                    output_dit[key] = value
            return output_dit
        elif isinstance(inputs, np.ndarray):
            return self.net.run(inputs)
        else:
            return None

    def get_all_tensor(self, input_data=None):
        tensors_dict = self.net.get_all_tensor()
        return tensors_dict

    def get_op_info(self):
        return self.net.op_info