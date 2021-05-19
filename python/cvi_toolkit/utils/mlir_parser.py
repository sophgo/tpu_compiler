import os
import sys
import mlir

class Operation:
    def __init__(self, op, body, idx):
        self.name = Operation.name(op)
        self.type = Operation.type(op)
        self.loc = Operation.loc(op)
        self.shape = Operation.shape(op)
        self.opds = Operation.operands(op, body, idx)

    def __str__(self):
        return self.name + "," + self.type + "," + self.loc + "," + str(self.shape) + "," + str(self.opds)

    @staticmethod
    def name(op):
        return mlir.ir.StringAttr(op.attributes['name']).value

    @staticmethod
    def type(op):
        return op.operation.name

    @staticmethod
    def loc(op):
        return op.get_asm().split('=')[0].strip('% ')

    @staticmethod
    def shape(op):
        shape_type = mlir.ir.ShapedType(op.operands[0].type)
        shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)]
        return shape

    @staticmethod
    def operands(op, body, idx):
        opds = []
        for opd in op.operands:
            for j in reversed(range(idx)):
                prev_op = body.operations[j]
                if prev_op.result == opd:
                    if Operation.type(prev_op) not in ['tpu.none', 'tpu.load_weight', 'tpu.weight_file']:
                        opds.append(Operation.name(prev_op))
        return opds

class MlirParser:
    def __init__(self, mlir_file):
        with open(mlir_file, 'r') as f:
            context = f.read()
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.m = mlir.ir.Module.parse(context, self.ctx)
        self.body = self.m.body.operations[0].regions[0].blocks[0]
        self.ops = []

    def get_all_ops(self):
        if len(self.ops) != 0:
            return self.ops

        for i in range(len(self.body.operations)):
            op = self.body.operations[i]
            type = Operation.type(op)
            if type in ['tpu.weight_file', 'tpu.load_weight',
                        'tpu.none', 'std.return']:
                continue
            self.ops.append(Operation(op, self.body, i))
        return self.ops

    def find_operations(self, op_type_name):
        ops = list(filter(lambda x: x.operation.name == op_type_name,
                      self.body.operations))
        return ops

    def get_weight_file_name(self):
        ops = self.find_operations('tpu.weight_file')
        if len(ops) == 0:
            return None
        weight_file_op = ops[0]
        return mlir.ir.StringAttr(weight_file_op.attributes['filename']).value

    def get_batch_size(self, idx):
        inputs = self.find_operations('tpu.input')
        assert(len(inputs) >= idx + 1)
        op = inputs[idx]
        shape_type = mlir.ir.ShapedType(op.operands[0].type)
        shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)]
        return shape[0]

    def get_output_op_names(self):
        return_op = self.find_operations('std.return')[0]
        outputs = []
        for op in self.body.operations:
            if op == return_op:
                continue
            for opd in return_op.operands:
                if op.result == opd:
                    name = mlir.ir.StringAttr(op.attributes['name']).value
                    outputs.append(name)
        return outputs

if __name__ == '__main__':
    parser = MlirParser(sys.argv[1])
    for op in parser.get_all_ops():
        print(op)