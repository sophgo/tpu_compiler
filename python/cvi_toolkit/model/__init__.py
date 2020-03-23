import pymlir

class CVI_MODEL(object):
    def __init__(self, register_model=None):
        self.cvi_model = pymlir.module()
        self.init_flag = False
        if register_model != None:
            self.NN_RegisterModel(register_model)
            self.init_flag = True

    def NN_RegisterModel(self, mlir_model):
        if self.init_flag:
            print("[WARNING] cvi model is already initialized")
        else:
            self.cvi_model.load(mlir_model)
            self.init_flag = True

    def NN_Forward(self, input):
        if self.init_flag:
            self.cvi_model.run(input)
        else:
            print("[ERROR] cvi model not initialize")

    def get_all_tensor(self):
        if self.init_flag:
            return self.cvi_model.get_all_tensor()
        else:
            print("[ERROR] cvi model not initialize")
            return None

    def get_op_info(self):
        if self.init_flag:
            return self.cvi_model.op_info
        else:
            print("[ERROR] cvi model not initialize")
            return None