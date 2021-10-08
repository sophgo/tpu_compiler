#!/usr/bin/python3
"""
Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
"""

from argparse import ArgumentParser
from cvi_toolkit.transform.BaseConverter import TensorType
from cvi_toolkit.transform.caffe_converter import CaffeConverter
from cvi_toolkit.utils.mlir_shell import *
import numpy as np

class MyCaffeConverter(CaffeConverter):
    def __init__(self, model_name, prototxt, caffe_model, mlir_file_path, batch_size=1):
        super().__init__(model_name, prototxt, caffe_model, mlir_file_path, batch_size)
        self.caffeop_factory['Python'] = lambda layer: self.convert_python_op(layer)


    def convert_python_op(self, layer):
        assert(self.layerType(layer) == "Python")
        op0, shape0, _ = self.getOperand(layer.bottom[0])
        op1, shape1, _ = self.getOperand(layer.bottom[1])
        operands = list()
        operands.append(op0)
        operands.append(op1)
        assert(shape0 == shape1)
        p = layer.python_param

        custom_op_param = {
            'tpu': True,
            'do_quant': True,
            'operation_name': p.layer,
            'threshold_overwrite': 'none',
            'param': {
            }
        }
        print("layer name: {}, top name: {}\n".format(layer.name, layer.top[0]))
        output_shape = list(shape0)
        custom_op = self.CVI.add_custom_op(layer.name,
                                           operands, output_shape, **custom_op_param)
        self.addOperand(layer.top[0], custom_op, output_shape, TensorType.ACTIVATION)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_dat", type=str)
    parser.add_argument("--mlir_file_path", type=str)
    args = parser.parse_args()
    tmp_mlir = args.mlir_file_path + "_tmp"
    c = MyCaffeConverter('mymodel', args.model_path, args.model_dat, tmp_mlir, batch_size=1)
    c.run()

    mlir_opt(tmp_mlir, args.mlir_file_path, "_op_info.csv")

    # generate random input for test
    shape = [1,3,160,250]
    data0 = np.random.rand(np.prod(shape))
    data0 = data0.reshape(shape).astype(np.float32)
    data1 = np.random.rand(np.prod(shape))
    data1 = data1.reshape(shape).astype(np.float32)
    np.savez("input.npz",**{"input0":data0,"input1":data1})

