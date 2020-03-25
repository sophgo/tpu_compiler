from .base_model import model_base
import pymlir



class MLIRModel(model_base):
    def __init__(self):
        self.net = pymlir.module()

    def load_model(self, mlirfile):
        self.net.load(mlirfile)

    def inference(self, input):
        self.net.run(input)
        output_op = self.net.op_info[-1]
        data = self.net.get_all_tensor()
        return data[output_op['name']]