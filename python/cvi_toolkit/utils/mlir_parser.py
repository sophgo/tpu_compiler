import os
import sys
import mlir

class MlirParser:
    def __init__(self, mlir_file):
        with open(mlir_file, 'r') as f:
            context = f.read()
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.m = mlir.ir.Module.parse(context, self.ctx)
        self.body = self.m.body.operations[0].regions[0].blocks[0]

    def get_operations(self, op_type_name):
        ops = list(filter(lambda x: x.operation.name == op_type_name,
                      self.body.operations))
        return ops

    def get_weight_file_name(self):
        ops = self.get_operations('tpu.weight_file')
        if len(ops) == 0:
            return None
        weight_file_op = ops[0]
        return mlir.ir.StringAttr(weight_file_op.attributes['filename']).value

    def get_batch_size(self, idx):
        inputs = self.get_operations('tpu.input')
        assert(len(inputs) >= idx + 1)
        op = inputs[idx]
        shape_type = mlir.ir.ShapedType(op.operands[0].type)
        shape = [shape_type.get_dim_size(i) for i in range(shape_type.rank)]
        return shape[0]

    def get_output_op_names(self):
        return_op = self.get_operations('std.return')[0]
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
    print(parser.get_weight_file_name())
    print(parser.get_output_op_names())
    print(parser.get_batch_size(0))