#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import subprocess
import numpy as np
import contextlib
import onnx
from cvi_toolkit.utils.mlir_shell import *
from cvi_toolkit.utils.intermediate_file import IntermediateFile


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)

class ModelTest(object):
    def __init__(self, chip_type, model_path, batch_size):
        self.chip_type = chip_type
        self.model_path = model_path
        self.batch_size = batch_size
        self.model_name = os.path.split(model_path)[-1].split(".")[0]
        self.fp32_mlir = self.model_name + ".mlir"
        self.cvimodel = self.model_name + ".cvimodel"
        self.input_path = "./input.npz"

    def __make_test_calibration_table__(self, table_name):
        blobs_interp_npz = IntermediateFile(self.model_name, 'full_precision_interp.npz', False)
        ret = mlir_inference(self.fp32_mlir, self.input_path, None, str(blobs_interp_npz))
        if ret != 0:
            raise RuntimeError("{} mlir inference failed".format(self.model_path))
        tensors = np.load(str(blobs_interp_npz))

        with open(table_name, "w") as f:
            for name in tensors:
                threshold = np.abs(np.max(tensors[name]))
                if np.isnan(threshold):
                    threshold = 10.0
                elif threshold >= 127.0:
                    threshold = 127.0
                elif threshold <= 0.001:
                    threshold = 1.0
                else:
                    pass
                f.write("{} {}\n".format(name, threshold))

    def run(self, quant_mode, input=None):
        if self.model_path.endswith(".onnx"):
            onnx_model = onnx.load(self.model_path)
            input_nodes = onnx_model.graph.input
            self.__gen_onnx_input__(input_nodes)
            transform_cmd = ['model_transform.py', '--model_type', 'onnx', '--model_name', self.model_name,
                             '--model_def', self.model_path, '--image', self.input_path, '--net_input_dims', '1,100',
                             '--tolerance', '0.99,0.99,0.99', '--mlir', self.fp32_mlir]
            subprocess.run(transform_cmd)
        elif self.model_path.endswith(".mlir"):
            tmp_mlir_file = IntermediateFile(self.model_name, 'fp32.mlir.tmp', False)
            op_info_csv = IntermediateFile(self.model_name, 'op_info.csv', True)
            ret = mlir_pseudo_weight(self.model_path, str(tmp_mlir_file))
            ret = mlir_opt(str(tmp_mlir_file), self.fp32_mlir, str(op_info_csv))
            if ret != 0:
                raise RuntimeError("{} opt failed".format(self.model_path))

        if "bf16" == quant_mode:
            deploy_cmd = ['model_deploy.py', '--model_name', self.model_name, '--mlir', self.fp32_mlir, '--all_bf16',
                          '--chip', self.chip_type, '--image', self.input_path, '--tolerance', '0.99,0.99,0.87',
                          '--correctness', '0.99,0.99,0.95', '--cvimodel', self.cvimodel]
        elif "int8" == quant_mode:
            # simple cali and convert to cvimodel
            table_file = IntermediateFile(self.model_name, 'calibration_table', True)
            self.__make_test_calibration_table__(str(table_file))
            deploy_cmd = ['model_deploy.py', '--model_name', self.model_name, '--mlir', self.fp32_mlir, '--calibration_table',
                          str(table_file), '--chip', self.chip_type, '--image',self. input_path, '--tolerance',
                          '0.10,0.10,0.1', '--correctness', '0.99,0.99,0.93', '--cvimodel', self.cvimodel]
        else:
            raise ValueError("Now just support bf16/int8")
        subprocess.run(deploy_cmd)

    def __gen_onnx_input__(self, input_nodes):
        self.input_data = {}
        for input in input_nodes:
            input_shape = []
            for i, dim in enumerate(input.type.tensor_type.shape.dim):
                if i == 0 and dim.dim_value <= 0 and self.batch_size != 0:
                    input_shape.append(self.batch_size)
                else:
                    input_shape.append(dim.dim_value)
            if 1 == input.type.tensor_type.elem_type:    # 1 for np.float32
                self.input_data[input.name] = np.random.randn(*input_shape).astype(np.float32)
            elif 7 == input.type.tensor_type.elem_type:  # 7 for np.int64 / torch.long
                self.input_data[input.name] = np.random.randint(0, 3, input_shape).astype(np.int64)
            elif 9 == input.type.tensor_type.elem_type:  # 9 for boolean
                self.input_data[input.name] = np.random.randint(0, 2, input_shape).astype(np.float32)
            else:
                raise ValueError("Not support now, add here")
            np.savez("input.npz", **self.input_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", help="model definition file.")
    parser.add_argument("--qmode", choices=['bf16', 'int8'], help="quant mode")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--chip_type", type=str, default="cv182x", help="chip type")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="tmp folder")
    # parser.add_argument("--excepts", default='-', help="excepts")
    # parser.add_argument("--graph", action='store_true', help="generate graph to pb file")
    args = parser.parse_args()

    if os.path.exists(args.tmp_dir):
         shutil.rmtree(args.tmp_dir)
    os.makedirs(args.tmp_dir)

    tmp_model_file = os.path.split(args.model_def)[-1]
    shutil.copy(args.model_def, os.path.join(args.tmp_dir, tmp_model_file))

    with pushd(args.tmp_dir):
        tool = ModelTest(args.chip_type, tmp_model_file, args.batch_size)
        tool.run(args.qmode)
