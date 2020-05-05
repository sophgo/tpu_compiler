class BaseConverter(object):
    def __init__(self):
        self.valueMap = dict() # {op_name: (mlir op, shape)}

    def init_importer(self):
        raise NotImplementedError('init_importer')

    def run(self):
        raise NotImplementedError('run')

    def addOperand(self, op_name, op, shape, tensor_type):
        self.valueMap[op_name] = (op, shape, tensor_type)

    def getOperand(self, op_name):
        return self.valueMap[op_name]