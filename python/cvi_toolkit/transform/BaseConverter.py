import numpy as np
from enum import Enum
from subprocess import check_call

class TensorType(Enum):
    ACTIVATION = 'ACTIVATION'
    TENSOR = 'TENSOR'
class BaseConverter(object):
    def __init__(self):
        self.valueMap = dict() # {op_name: (mlir op, shape)}
        self.shape_info = dict()

    def checkShape(self, op_name, trg_shape):
        try:
            ref_shape = self.shape_info[op_name]
            if (np.array(ref_shape) != np.array(ref_shape)).any():
                raise ValueError("Shape mismatch: onnx {} {} vs mlir {}"\
                    .format(op_name, ref_shape, trg_shape))
        except KeyError:
            print("Can't get {} 's shape_info sikpped.".format(op_name))

    def init_importer(self):
        raise NotImplementedError('init_importer')

    def run(self):
        raise NotImplementedError('run')

    def addOperand(self, op_name, op, shape, tensor_type):
        if isinstance(op_name, int):
            op_name = str(op_name)
        if len(self.shape_info) > 0 and tensor_type == TensorType.ACTIVATION:
            self.checkShape(op_name, shape)
        self.valueMap[op_name] = (op, shape, tensor_type)

    def getOperand(self, op_name):
        if isinstance(op_name, int):
            op_name = str(op_name)
        return self.valueMap[op_name]