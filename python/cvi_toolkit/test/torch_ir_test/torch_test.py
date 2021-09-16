#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torchvision.models as models
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.model.mlir_model import MLIRModel
from cvi_toolkit.utils.mlir_shell import mlir_quant, \
     mlir_opt, mlir_to_cvimodel, run_cvimodel
from cvi_toolkit.numpy_helper import npz_compare
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime
import pyruntime
import numpy as np
import os
import sys
import gc
import re

TEST_ONNX_IR = [
    "Add",
    "Conv2d", # Conv with 2d case
    "Std",
    "Squeeze",
    "Linear",
]

def cvimodel_inference(inputs, model_name):
    model = pyruntime.Model(model_name)
    assert(model != None)
    if isinstance(inputs, dict):
        for i, input in enumerate(inputs.values()):
            model_in = model.inputs[i].data
            assert(model_in.dtype == input.dtype)
            assert(model_in.size == input.size)
            model_in[:] = input.reshape(model_in.shape)
    else:
        model_in = model.inputs[0].data
        model_in[:] = inputs.reshape(model_in.shape)

    model.forward()
    outputs = {}
    for output in model.outputs:
        outputs[output.name] = np.array(output.data)
    return outputs

def get_chip_name():
    runchip = os.environ.get('SET_CHIP_NAME', None)
    if not runchip:
        log.warning(
            "no found SET_CHIP_NAME environment value, set 183x as default")
        return "cv183x"
    return runchip

def make_test_calibration_table(tensors, table_name):
    QUANT_BITWIDTH = {}
    # simple calibration table
    with open(table_name, 'w') as f:
        for name in tensors:
            t = 1.1 * max(np.abs(tensors[name].flatten())) + 0.01
            f.write("{} {}\n".format(name, t))
        for key,value in QUANT_BITWIDTH.items():
            f.write("bitwidth {} {}\n".format(key, value))

def _fill_inputs(ort_session, inputs):
    inodes = ort_session.get_inputs()
    if len(inodes) == 1:
        dtype = np.int64 if inodes[0].type == 'tensor(int64)' \
            else np.float32
        return {inodes[0].name: inputs.astype(dtype)}
    # inputs is map
    assert(len(inodes) == len(inputs))
    data = {}
    for i in range(len(inodes)):
        name = inodes[i].name
        dtype = np.int64 if inodes[i].type == 'tensor(int64)' \
                            else np.float32
        data[name] = inputs[name].astype(dtype)
    return data

def _onnx_inference(inputs, model_name, input_name="input", input_cb=None):
    ort_session = onnxruntime.InferenceSession(model_name)
    if callable(input_cb):
        ort_inputs = input_cb(model_name, "onnx", input)
    else:
        ort_inputs = _fill_inputs(ort_session, inputs)

    outs = ort_session.run(None, ort_inputs)
    ort_outputs = ort_session.get_outputs()
    outputs = {}
    idx = 0
    for output in ort_outputs:
        outputs[output.name] = outs[idx]
        idx = idx + 1
    return outputs

def onnx_inference(input, model_def, input_cb = None):
    return _onnx_inference(input, model_def, input_cb=input_cb)

NOT_SUPPORT_CMDBUF_TEST_IR = [""]
NOT_SUPPORT_BF16_TEST_IR = [""]
NOT_SUPPORT_INT8_TEST_IR = [""] # just for save test time

class TORCH_IR_TESTER(object):
    def __init__(self):
        self.converter = None
        self.cvi_model_test = True

        self.test_function = {
            "Add": self.test_Add,
            "Conv2d": self.test_Conv2d,
            "Linear": self.test_Linear,
            "Std": self.test_Std,
            "Squeeze": self.test_Squeeze,
        }
        self.set_quant_mode()

    def set_quant_mode(self, mode="int8"):
        if mode == "int8":
            self.quant_mode = "int8"
        elif mode == "bf16":
            self.quant_mode = "bf16"
        else:
            raise RuntimeError("Not support quant mode {}".format(mode))

    def onnx_convert_and_infernece(self, input_data, model_name, torch_output, input_cb=None):
        fp32_mlir = "{}.mlir".format(model_name)
        model_def = model_name + '.onnx'
        converter = OnnxConverter(model_name, model_def, fp32_mlir, batch_size=input_data.shape[0])
        converter.run()
        del converter
        gc.collect()

        input_data = input_data.data.numpy().astype(np.float32)
        onnx_outs = onnx_inference(input_data, model_def, input_cb)
        num_outputs = len(onnx_outs)

        ##test pytorch out_data between onnx out_data
        if num_outputs == 1:
            onnx_out = list(onnx_outs.values())[0]
            np.testing.assert_allclose(torch_output.flatten(), onnx_out.flatten(), rtol=1e-5, atol=1e-01)

        input_npz = "{}_input_fp32.npz".format(model_name)
        np.savez(input_npz, input=input_data)

        fp32_opt_mlir = "{}_opt.mlir".format(model_name)
        fp32_csv = "{}_fp32.csv".format(model_name)
        mlir_opt(fp32_mlir, fp32_opt_mlir, fp32_csv)
        mlir_model = None
        mlir_model = MLIRModel()
        mlir_model.load_model(fp32_opt_mlir)
        mlir_outs = mlir_model.inference(input_data)
        fp32_tensors = mlir_model.get_all_tensor()

        assert(len(mlir_outs) == num_outputs)
        if num_outputs > 1:
            patten = re.compile(r"_[A-Z]\w+?$")
            for name in mlir_outs:
                onnx_name = patten.sub("", name)
                print("Compare mlir[{}] : onnx[{}]".format(name, onnx_name))
                np.testing.assert_allclose(mlir_outs[name].flatten(), onnx_outs[onnx_name].flatten(), rtol=1e-5, atol=1e-01)
        else:
            if isinstance(mlir_outs, dict):
                mlir_outs = list(mlir_outs.values())[0]
            onnx_out = onnx_outs.popitem()[1]
            np.testing.assert_allclose(mlir_outs.flatten(), onnx_out.flatten(), rtol=1e-5, atol=1e-01)

        mlir_npz = "{}_fp32.npz".format(model_name)
        np.savez(mlir_npz, **fp32_tensors)

        quant_mode = "int8"
        tensors = mlir_model.get_all_tensor()
        if quant_mode == "int8":
            for i in NOT_SUPPORT_INT8_TEST_IR:
                if i == model_name:
                    print("{} not support bf16 test!".format(model_name))
            table_name = "{}_cali_table".format(model_name)
            # gen cali table
            make_test_calibration_table(tensors, table_name)

            # quant
            quant_mlir = "{}_quant_int8.mlir".format(model_name)
            int8_csv = "{}_int8.csv".format(model_name)
            chip = get_chip_name()
            ret = mlir_quant(fp32_opt_mlir, quant_mlir, chip,
                                int8_csv, calib_table=table_name)
            if ret < 0: raise RuntimeError("tpu_quant failed")

            # get mlir output
            del mlir_model
            mlir_model = MLIRModel()
            mlir_model.load_model(quant_mlir)
            mlir_int8_outs = mlir_model.inference(input_data)
            assert(len(mlir_int8_outs) == num_outputs)
            int8_tensors = mlir_model.get_all_tensor()
            ref_npz = "{}_all_tensor_int8_mlir.npz".format(model_name)
            np.savez(ref_npz, **int8_tensors)
            npz_compare([ref_npz, mlir_npz,  "--tolerance",
                            "0.6,0.6,0.6", "--dequant", "--op_info", int8_csv])

            # gen cvimodel
            cvimodel = "{}_int8.cvimodel".format(model_name)
            ret = mlir_to_cvimodel(quant_mlir, cvimodel)
            if ret < 0: raise RuntimeError("gen_cvimodel failed")

            # run cvi_model
            output_tensor_npz = "{}_all_tensor_int8_cvi.npz".format(model_name)

            if isinstance(input_data, dict):
                count = 0
                input_int = {}
                for key, value in input_data.items():
                    if key+'_quant_i8' in int8_tensors:
                        input_int['input'+str(count)] = int8_tensors[key+'_quant_i8'].astype(np.int8)
                    elif key+'_quant_u16' in int8_tensors:
                        input_int['input'+str(count)] = int8_tensors[key+'_quant_u16'].astype(np.uint16)
                    count += 1
            else:
                input_int = None
                if 'input_quant_i8' in int8_tensors:
                    input_int = int8_tensors['input_quant_i8'].astype(np.int8)
                elif 'input_quant_u16' in int8_tensors:
                    input_int = int8_tensors['input_quant_u16'].astype(np.uint16)
                else:
                    input_int = input_data
            cvi_outs = cvimodel_inference(input_int, cvimodel)
            assert(len(cvi_outs) == num_outputs)
            for name in cvi_outs:
                if name not in int8_tensors:
                    raise RuntimeError("cvimodel output name not correct")
            np.savez(output_tensor_npz, **cvi_outs)
            npz_compare([output_tensor_npz, ref_npz,
                            "--tolerance", "0.99,0.99,0.9"])

    def pytorch_transform_onnx(self, model, input_data, test_onnx_name):
        # Create some sample  input in the shape this model expects
        input_names = ['input']
        output_names = ['output']
        onnx_name = test_onnx_name+'.onnx'
        torch.onnx.export(model,
            input_data,
            onnx_name,
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=input_names,
            output_names=output_names,)

    def test_Add(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.predict = torch.nn.Linear(1, 1)

            def forward(self, x):
                x = torch.add(x, x)
                return x

        input_shape = [1, 3, 8, 8]
        test_onnx_name = 'Add'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Std(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                mean = x.mean(-1).unsqueeze(-1)
                std = torch.std(x, -1).unsqueeze(-1)
                return (x - mean) / (std + 0.0001)

        input_shape = [1, 3, 32, 1024]
        test_onnx_name = 'Std'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Squeeze(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.negative(x)
                x = torch.squeeze(x)
                x = torch.add(x,x)
                return x

        input_shape = [3, 1, 8, 1]
        test_onnx_name = 'Squeeze'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Linear(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(10,20,bias=False)

            def forward(self, x):
                x = self.linear(x)
                return x

        input_shape = [3, 24, 10]
        test_onnx_name = 'Linear'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1], input_shape[2])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Conv2d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv_test = nn.Conv2d(in_channels=3,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            padding=1)

            def forward(self, x):
                x = self.conv_test(x)
                return x

        batch_size = 1
        test_onnx_name = 'Conv2d'
        ##torch needn't weight and bias
        input_data_np = np.random.randn(batch_size, 3, 27, 27).astype(np.float32)
        input_data = torch.from_numpy(input_data_np).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

if __name__ == "__main__":
    os.makedirs("torch_test", exist_ok=True)
    os.chdir("torch_test")
    tester = TORCH_IR_TESTER()
    if len(sys.argv) >= 3:
        input_shape = sys.argv[1].split(",")
        if len(sys.argv) >= 4:
            tester.test_model(input_shape, sys.argv[2], sys.argv[3])
        else:
            tester.test_model(input_shape, sys.argv[2])
        exit(0)
    elif len(sys.argv) == 2:
        tester.test_function.get(sys.argv[1])()
        exit(0)
    elif len(sys.argv) == 1:
        pass_list_i8 = list()
        pass_list_bf16 = list()

        for i in TEST_ONNX_IR:
            if i not in NOT_SUPPORT_INT8_TEST_IR:
                tester.test_function.get(i)()
                pass_list_i8.append(i)
                print("TEST {} Finish".format(i))

        for i in TEST_ONNX_IR:
            if i not in NOT_SUPPORT_BF16_TEST_IR:
                tester.set_quant_mode(mode="bf16")
                tester.test_function.get(i)()
                pass_list_bf16.append(i)

        print("INT8 {} PASS {}".format("="*4, "="*4))
        for i in pass_list_i8:
            if i not in NOT_SUPPORT_INT8_TEST_IR:
                print("\t {}".format(i))

        print("BF16 {} PASS {}".format("="*4, "="*4))
        for i in pass_list_bf16:
            if i not in NOT_SUPPORT_BF16_TEST_IR:
                print("\t {}".format(i))

    else:
        print("Usage: exe.py [input_shape] [model]")
        exit(-1)


