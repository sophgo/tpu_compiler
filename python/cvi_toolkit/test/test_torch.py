#!/usr/bin/python3
# -*- coding: utf-8 -*-

from cvi_toolkit.model.mlir_model import MLIRModel
from cvi_toolkit.utils.mlir_shell import mlir_quant, \
     mlir_opt, mlir_to_cvimodel, run_cvimodel
from cvi_toolkit.numpy_helper import npz_compare
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.numpy_helper.npz_compare import fp32_to_bf16
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime
import pyruntime
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import sys
import gc
import re

TEST_TORCH_IR = [
    "Activation",
    "AvgPool",
    "AdaptiveAvgPool2d",  #  input_size % output_size == 0
    # "Bilinear", ## Bilinear not support
    "Batch_Norm", ##Batch_norm_2d and Instance_Norm_2d easily will fail
    "Conv3d", # sunpport Conv with 3d, 2d, 1d case
    "ConvTranspose",
    "Cat_Chunk",
    "Clip",
    "ConstantPad",
    # "ChannelShuffle", ## ChannelShuffle not support
    # "Dropout", ## Dropout not support
    "Expand",
    "Flatten", ## Unflatten not support
    "GRU",
    "Identity",
    "Log",
    "LSTM",
    "LayerNorm",
    "Linear",
    "LogSigmoid", #some times fail
    "LogSoftmax",
    # "Mulit_attention_api",  ## now not support
    "MaxPool", ## Maxpool_1d and Max_Un_Pool2d not support
    "MaxPool3d",
    "Max_Min",
    "Math", ## sum, prod not support
    "masked_fill",
    "Norm",
    "Multi_input",
    "Pow",
    "Repeat",   ## repeat_interleave nonx not support
    "ReflectionPad", ## ReflectionPad_2d not support
    "Std",
    "Squeeze",
    "Size",
    "Sum",
    "Scale",
    "SiLU",
    # "Unfold", ##Unfold not support
    "Upsample",
    "ZeroPad2d",
]

NOT_SUPPORT_CMDBUF_TEST_IR = [""]
NOT_SUPPORT_BF16_TEST_IR = [""]
NOT_SUPPORT_INT8_TEST_IR = ["masked_fill"] # just for save test time

def cvimodel_inference(inputs, model_name):
    model = pyruntime.Model(model_name)
    for i in model.inputs:
        name = i.name
        data = inputs[name]
        if name.endswith('_quant_i8'):
            data = data.astype(np.int8)
        elif name.endswith('_quant_u16'):
            data= data.astype(np.uint16)
        elif name.endswith('_quant_i16'):
            data = data.astype(np.int16)
        elif name.endswith('_quant_bf16'):
            data = fp32_to_bf16(data)
        i.data[:] = data.reshape(i.data.shape)
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
    # simple calibration table
    with open(table_name, 'w') as f:
        for name in tensors:
            t = 1.1 * max(np.abs(tensors[name].flatten())) + 0.01
            f.write("{} {}\n".format(name, t))

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

class TORCH_IR_TESTER(object):
    def __init__(self):
        self.converter = None
        self.cvi_model_test = True

        self.test_function = {
            "Activation": self.test_Activation,
            "AvgPool": self.test_AvgPool,
            "AdaptiveAvgPool2d": self.test_AdaptiveAvgPool2d,
            "Batch_Norm": self.test_Batch_Norm,
            "Bilinear": self.test_Bilinear,
            "Conv3d": self.test_Conv3d,
            "ConvTranspose": self.test_ConvTranspose,
            "Cat_Chunk": self.test_Cat_Chunk,
            "Clip": self.test_Clip,
            "ConstantPad": self.test_ConstantPad,
            "ChannelShuffle": self.test_ChannelShuffle,
            "Dropout": self.test_Dropout,
            "Expand": self.test_Expand,
            "Flatten": self.test_Flatten,
            "GRU": self.test_GRU,
            "Identity": self.test_Identity,
            "Log": self.test_Log,
            "LogSigmoid": self.test_LogSigmoid,
            "LogSoftmax": self.test_LogSoftmax,
            "LayerNorm": self.test_LayerNorm,
            "Linear": self.test_Linear,
            "LSTM": self.test_LSTM,
            "MaxPool": self.test_MaxPool,
            "MaxPool3d": self.test_MaxPool3d,
            "Math": self.test_Math,
            "masked_fill": self.test_masked_fill,
            "Mulit_attention_api": self.test_Mulit_attention_api,
            "Multi_input": self.test_Multi_input,
            "Max_Min": self.test_Max_Min,
            "Norm": self.test_Norm,
            "Pow": self.test_Pow,
            "Repeat": self.test_Repeat,
            "ReflectionPad": self.test_ReflectionPad,
            "Std": self.test_Std,
            "Scale": self.test_Scale,
            "Squeeze": self.test_Squeeze,
            "Size": self.test_Size,
            "SiLU": self.test_SiLU,
            "Sum": self.test_Sum,
            "Unfold": self.test_Unfold,
            "Upsample": self.test_Upsample,
            "ZeroPad2d": self.test_ZeroPad2d,
        }
        self.set_quant_mode()

    def set_quant_mode(self, mode="int8"):
        if mode == "int8":
            self.quant_mode = "int8"
        elif mode == "bf16":
            self.quant_mode = "bf16"
        else:
            raise RuntimeError("Not support quant mode {}".format(mode))

    def onnx_convert_and_infernece(self, inputs, model_name, torch_output):
        fp32_mlir = "{}.mlir".format(model_name)
        model_def = model_name + '.onnx'
        converter = OnnxConverter(model_name, model_def, fp32_mlir, batch_size=0)
        converter.run()
        del converter
        gc.collect()
        input_npz = "{}_in_fp32.npz".format(model_name)
        if isinstance(inputs, tuple):
            input_data = {}
            for i in range(len(inputs)):
                key = "in_{}".format(i)
                input_data[key] = inputs[i].data.numpy().astype(np.float32)
            np.savez(input_npz, **input_data)
        else:
            input_data = inputs.data.numpy().astype(np.float32)
            np.savez(input_npz, input=input_data)
        onnx_outs = onnx_inference(input_data, model_def)
        num_outputs = len(onnx_outs)

        ##test pytorch out_data between onnx out_data
        if num_outputs == 1:
            onnx_out = list(onnx_outs.values())[0]
            np.testing.assert_allclose(torch_output.flatten(), onnx_out.flatten(), rtol=1e-5, atol=1e-01)
        else:
            assert(len(torch_output) == num_outputs)
            keys = list(onnx_outs)
            for i in range(num_outputs):
                print("==> Torch vs Onnx, at[{}]".format(i))
                np.testing.assert_allclose(torch_output[i].data.numpy().flatten(), onnx_outs[keys[i]].flatten(), rtol=1e-5, atol=1e-01)

        fp32_opt_mlir = "{}_opt.mlir".format(model_name)
        fp32_csv = "{}_fp32.csv".format(model_name)
        mlir_opt(fp32_mlir, fp32_opt_mlir, fp32_csv)
        self.mlir_model = None
        self.mlir_model = MLIRModel()
        self.mlir_model.load_model(fp32_opt_mlir)
        mlir_outs = self.mlir_model.inference(input_data)
        fp32_tensors = self.mlir_model.get_all_tensor()

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
        print("Compare Torch and Onnx success")

        mlir_npz = "{}_fp32.npz".format(model_name)
        np.savez(mlir_npz, **fp32_tensors)

        tensors = self.mlir_model.get_all_tensor()
        if self.quant_mode == "int8":
            for i in NOT_SUPPORT_INT8_TEST_IR:
                if i == model_name:
                    print("{} not support int8 test!".format(model_name))
            table_name = "{}_cali_table".format(model_name)
            # gen cali table
            make_test_calibration_table(tensors, table_name)

            # quant
            quant_mlir = "{}_quant_int8.mlir".format(model_name)
            int8_csv = "{}_int8.csv".format(model_name)
            chip = get_chip_name()
            ret = mlir_quant(fp32_opt_mlir, quant_mlir, chip,
                                int8_csv, calib_table=table_name, quantize="int8")
            if ret < 0: raise RuntimeError("tpu_quant failed")

            # get mlir output
            del self.mlir_model
            self.mlir_model = MLIRModel()
            self.mlir_model.load_model(quant_mlir)
            mlir_int8_outs = self.mlir_model.inference(input_data)
            assert(len(mlir_int8_outs) == num_outputs)
            int8_tensors = self.mlir_model.get_all_tensor()
            ref_npz = "{}_all_tensor_int8_mlir.npz".format(model_name)
            np.savez(ref_npz, **int8_tensors)
            npz_compare([ref_npz, mlir_npz,  "--tolerance",
                            "0.6,0.6,0.5", "--dequant", "--op_info", int8_csv])
            # gen cvimodel
            cvimodel = "{}_int8.cvimodel".format(model_name)
            ret = mlir_to_cvimodel(quant_mlir, cvimodel, inputs_type="SAME", outputs_type="FP32")
            if ret < 0: raise RuntimeError("gen_cvimodel failed")

            # run cvi_model
            output_tensor_npz = "{}_all_tensor_int8_cvi.npz".format(model_name)

            cvi_outs = cvimodel_inference(int8_tensors, cvimodel)
            assert(len(cvi_outs) == num_outputs)
            for name in cvi_outs:
                if name not in int8_tensors:
                    raise RuntimeError("cvimodel output name not correct")
            np.savez(output_tensor_npz, **cvi_outs)
            npz_compare([output_tensor_npz, ref_npz,
                            "--tolerance", "0.99,0.99,0.90"])
        elif self.quant_mode == "bf16":
            for i in NOT_SUPPORT_BF16_TEST_IR:
                if i == model_name:
                    print("{} not support bf16 test!".format(model_name))
                    return
            # opt
            fp32_opt_mlir = "{}_opt_bf16.mlir".format(model_name)
            fp32_csv = "{}_fp32.csv".format(model_name)
            mlir_opt(fp32_mlir, fp32_opt_mlir, fp32_csv)

            bf16_csv = "{}_bf16.csv".format(model_name)

            # quant
            quant_mlir = "{}_quant_bf16.mlir".format(model_name)
            chip = get_chip_name()
            ret = mlir_quant(fp32_opt_mlir, quant_mlir, chip,
                            bf16_csv, quantize="bf16")
            if ret < 0: raise RuntimeError("tpu_quant failed")

            # get mlir output
            del self.mlir_model
            self.mlir_model = MLIRModel()
            self.mlir_model.load_model(quant_mlir)
            mlir_bf16_outs = self.mlir_model.inference(input_data)
            assert(len(mlir_bf16_outs) == num_outputs)
            bf16_tensors = self.mlir_model.get_all_tensor()
            ref_npz = "{}_all_tensor_bf16_mlir.npz".format(model_name)
            np.savez(ref_npz, **bf16_tensors)
            npz_compare([ref_npz, mlir_npz,  "--tolerance",
                        "0.8,0.8,0.8", "--dequant", "--op_info", bf16_csv])

            # gen cvimodel
            cvimodel = "{}_bf16.cvimodel".format(model_name)
            ret = mlir_to_cvimodel(quant_mlir, cvimodel, inputs_type="SAME", outputs_type="FP32")
            if ret < 0: raise RuntimeError("gen_cvimodel failed")

            # run cvi_model
            output_tensor_npz = "{}_all_tensor_bf16_cvi.npz".format(model_name)
            cvi_outs = cvimodel_inference(bf16_tensors, cvimodel)
            assert(len(cvi_outs) == num_outputs)
            for name in cvi_outs:
                if name not in bf16_tensors:
                    raise RuntimeError("cvimodel output name not correct")
            np.savez(output_tensor_npz, **cvi_outs)
            npz_compare([output_tensor_npz, ref_npz, "--op_info", bf16_csv, "--tolerance", "0.9,0.9,0.90", "-vv"])

        del self.mlir_model


    def pytorch_transform_onnx(self, model, inputs, test_name):
        in_names = []
        if isinstance(inputs, tuple):
            for i in range(len(inputs)):
                in_names.append("in_{}".format(i))
        else:
            in_names = ["in_0"]
        torch.onnx.export(model,
                          inputs,
                          test_name + ".onnx",
                          export_params=True,
                          opset_version=11,
                          verbose=True,
                          input_names=in_names)

    def test_LSTM(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=True)

            def forward(self, x, h_0, c_0):
                Y,(Y_h, Y_c) = self.rnn(x, (h_0, c_0))
                return Y,Y_h,Y_c

        test_name = 'LSTM'
        input = torch.randn(81, 1, 100)
        h_0 = torch.randn(2, 1, 128)
        c_0 = torch.randn(2,1,128)
        net = Net()
        outputs = net(input, h_0, c_0)

        # Use the exporter from  torch to convert to onnx
        inputs = (input, h_0, c_0)
        self.pytorch_transform_onnx(net, inputs, test_name)
        self.onnx_convert_and_infernece(inputs, test_name, outputs)

    def test_Bilinear(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.Bilinear = nn.Bilinear(20, 30, 40)

            def forward(self, x, y):
                ## input_shape = (100, 20), (100, 30)
                out = self.Bilinear(x, y) ## output_shape = (100, 40)
                return out

        input_data = {}
        input_shape = [100, 20, 30]
        input_data['input'] = torch.randn(input_shape[0], input_shape[1])
        input_data['input1'] = torch.randn(input_shape[0], input_shape[2])
        test_name = 'Bilinear'

        net = Net()
        torch_output_data = net(input_data['input'], input_data['input1'])

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Log(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.log(x)
                return x

        input_shape = [1, 3, 100, 100]
        test_name = 'Log'

        net = Net()
        input_data = torch.clamp(torch.randn(*input_shape), 8.0, 10.0)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_LogSigmoid(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(100, 200, bias=False)
                self.act = nn.LogSigmoid()

            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                return x

        input_shape = [3, 100, 100]
        test_name = 'LogSigmoid'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_LogSoftmax(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(32, 72, bias=False)
                self.act = nn.LogSoftmax(dim = 2)

            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                return x

        input_shape = [3, 100, 32]
        test_name = 'LogSoftmax'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_GRU(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.gru = nn.GRU(input_size=100, hidden_size=50)

            def forward(self, x):
                out, hidden = self.gru(x)
                return out

        test_name = 'GRU'
        input_shape = [8, 50, 100]
        input_data = torch.randn(input_shape)

        net = Net()
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Expand(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = x.expand(3, 4, 100, 100)
                return x

        input_shape = [3, 1, 100, 100]
        test_name = 'Expand'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    # Unfold + matmul + fold = Conv2d
    def test_Unfold(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.unfold = torch.nn.Unfold(kernel_size=(2, 2), stride=2)
                self.fold = torch.nn.Fold(output_size=(4, 4), kernel_size=(2, 2), stride=2)

            def forward(self, x):
                ##shape (N, C*Kn, L)
                ##Kn = C*kernel_size[0]*kernel_size[1]
                ##L = ((h + 2*padding - dilation*(kernel-1))/stride + 1) * (w + 2*padding - dilation*(kernel-1))/stride + 1
                x = self.unfold(x)
                x = self.fold(x)
                return x

        input_shape = [1, 2, 4, 4]
        test_name = 'Unfold'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Flatten(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.Unflatten = nn.Unflatten(1, torch.Size([2, 5, 5]))

            def forward(self, x):
                x = torch.flatten(x, start_dim=1, end_dim=3)
                # x = self.Unflatten(x)
                return x

        input_shape = [4, 1, 5, 10]
        test_name = 'Flatten'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_AdaptiveAvgPool2d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.AdptAvp = nn.AdaptiveAvgPool2d((4, 4))

            def forward(self, input):
                return self.AdptAvp(input)

        input_shape = [1, 32, 20, 20]
        test_name = 'AdaptiveAvgPool2d'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_SiLU(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.SiLU = nn.SiLU()

            def forward(self, x):
                return  self.SiLU(x)

        input_shape = [1, 32, 20, 20]
        test_name = 'SiLU'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_ChannelShuffle(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.channel_shuffle = nn.ChannelShuffle(2)

            def forward(self, x):
                x = self.channel_shuffle(x)
                return x

        input_shape = [1, 4, 100, 100]
        test_name = 'ChannelShuffle'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Upsample(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                ## input_shape = (n, c*r*r, h, w) --> (n, c, r*w, r*h)
                self.PixelShuffle = nn.PixelShuffle(3)
                self.Upsample_nearest = nn.Upsample(scale_factor=2, mode='nearest')
                self.Upsample_nearest_new = nn.UpsamplingNearest2d(scale_factor=2)
                self.Upsample_bilinear = nn.Upsample(scale_factor=2, mode='bilinear')
                self.Upsample_bilinear_new = nn.UpsamplingBilinear2d(scale_factor=2)

            def forward(self, x):
                x = self.PixelShuffle(x)
                x = self.Upsample_nearest_new(x)
                x = self.Upsample_bilinear_new(x)
                return x

        input_shape = [1, 9, 100, 100]
        test_name = 'Upsample'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)


    def test_Identity(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.Identity = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)

            def forward(self, x):
                x = self.Identity(x)
                return x

        input_shape = [1, 3, 100, 100]
        test_name = 'Identity'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Max_Min(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                max_x, e = torch.max(x, dim=3, keepdim=True)
                min_x, e = torch.min(x, dim=3, keepdim=True)
                x = torch.add(max_x, min_x)
                return x

        input_shape = [3, 1, 100, 100]
        test_name = 'Max_Min'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Repeat(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = x.repeat(2, 6)
                # x = torch.repeat_interleave(x, repeats=4, dim=1)
                return x

        input_shape = [1, 3, 100, 100]
        test_name = 'Repeat'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Squeeze(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.squeeze(x)
                x = torch.add(x,x)
                return x

        input_shape = [3, 2, 256, 256]
        test_name = 'Squeeze'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Dropout(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.dropout = nn.Dropout(p=0.5)

            def forward(self, x):
                x = self.dropout(x)
                return x

        input_shape = [1, 3, 100, 100]
        test_name = 'Dropout'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Math(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.exp(x)
                x = torch.add(x, x)
                x = torch.min(x, x+1)
                x = torch.max(x, x+1)
                # x = torch.prod(x)
                # x = torch.sum(x)
                return x

        input_shape = [1, 4, 100, 100]
        test_name = 'Math'

        net = Net()
        input_data = torch.randn(*input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Cat_Chunk(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.div_num = 2

            def forward(self, x):
                y = torch.negative(x)*2
                x = torch.cat((x, y), 1)
                x = torch.chunk(x, self.div_num, dim=1)
                x = torch.negative(x[0])
                return x

        input_shape = [1, 3, 100, 100]
        test_name = 'Cat_Chunk'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Sum(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.sum(x, 3)
                return x

        input_shape = [1, 3, 8, 8]
        test_name = 'Sum'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Size(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self,x):
                y = torch.ones(x.size(1))
                return torch.add(x, y)

        input_shape = [100, 256]
        test_name = "Size"

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_masked_fill(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self,x):
                y = x.masked_fill(x==0, value=torch.tensor(-50.0))
                z = x.masked_fill(x!=0, value=torch.tensor(1.0))
                return y + z

        input_shape = [2, 3, 100]
        test_name = "masked_fill"

        net = Net()
        input_data = torch.randint(0, 1000, input_shape)
        input_data[:,:,70:] = 0
        torch_output_data = net(input_data)
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Std(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                mean = x.mean(-1).unsqueeze(-1)
                std = torch.std(x, -1).unsqueeze(-1)
                return (x - mean) / (std + 0.0001)

        input_shape = [1, 3, 100, 100]
        test_name = 'Std'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Scale(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.pool = nn.AvgPool2d((64,32), stride=1)
                self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                x1 = self.pool(x)
                x2 = self.conv1(x)
                x3 = x2 * x1
                x4 = self.conv2(x3)
                return x4

        input_shape = [8, 64, 64, 32]
        test_name = 'Scale'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Mulit_attention_api(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.embedding_layer = torch.nn.Embedding(3, 6)
                self.multihead_attn = nn.MultiheadAttention(6, 2)

            def forward(self, x):
                x = self.embedding_layer(x) ##shape: [1, 3, 6]
                attn_output, attn_output_weights = self.multihead_attn(x, x, x)
                return attn_output

        test_name = 'Mulit_attention_api'
        input_shape = [1, 3]
        input_data = torch.randn(input_shape)
        input_data1 = torch.LongTensor([[0, 1, 2]]) ##shape: [1, 3]

        net = Net()
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Norm(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.norm(x, p=2, dim=2, keepdim=True)
                return x

        input_shape = [3, 10, 100, 100]
        test_name = 'Norm'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Pow(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv2d = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                x1 = torch.pow(x, 3)
                x2 = self.conv2d(x)
                x2 = torch.relu(x2)
                x3 = x1 + x2
                return x3

        input_shape = [3, 10, 50, 50]
        test_name = 'Pow'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Multi_input(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv_1d = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3)
                self.layer_norm = nn.LayerNorm(98)
                self.rnn = nn.LSTM(input_size=98, hidden_size=128, bidirectional=True)

            def forward(self, x, h_0, c_0):
                x = self.conv_1d(x)
                x = self.layer_norm(x)
                Y,(Y_h, Y_c) = self.rnn(x, (h_0, c_0))
                return Y,Y_h,Y_c

        test_name = 'Multi_input'
        net = Net()
        input_data = torch.randn(81, 3, 100)
        h_0 = torch.randn(2, 1, 128)
        c_0 = torch.randn(2, 1, 128)
        outputs = net(input_data, h_0, c_0)

        # Use the exporter from  torch to convert to onnx
        inputs = (input_data, h_0, c_0)
        self.pytorch_transform_onnx(net, inputs, test_name)
        self.onnx_convert_and_infernece(inputs, test_name, outputs)

    def test_Linear(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(10,20,bias=False)

            def forward(self, x):
                x = torch.transpose(x, 1, 2)
                x = torch.squeeze(x)
                x = torch.add(x,x)
                return x

        input_shape = [3, 100, 100]
        test_name = 'Linear'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1], input_shape[2])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_LayerNorm(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layer_norm = nn.LayerNorm([24, 50])

            def forward(self, x):
                x = self.layer_norm(x)
                return x

        input_shape = [3, 24, 50]
        test_name = 'LayerNorm'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Activation(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(100, 200, bias=False)
                self.softplus = nn.Softplus()
                self.hardsigmoid = nn.Hardsigmoid()
                self.prelu = nn.PReLU()
                self.ReLU6 = nn.ReLU6(inplace=True)
                self.mish = nn.Mish()
                self.Softmax = nn.Softmax(dim=1)
                self.Softmax_2d = nn.Softmax2d()

            def forward(self, input):
                #tanh
                x = self.linear(input)
                y0 = torch.tanh(x)
                ##relu
                x = self.linear(input)
                y2 = torch.relu(x)
                ##sigmoid
                x = self.linear(input)
                y1 = torch.sigmoid(x)
                ##leaky_relu
                # x = self.linear(input)
                # y3 = F.leaky_relu(x)
                ##elu
                x = self.linear(input)
                y4 = F.elu(x)
                ##softplus
                x = self.linear(input)
                y5 = self.prelu(x)
                ##hardsigmoid
                x = self.linear(input)
                y6 = self.hardsigmoid(x)
                ##prelu
                x = self.linear(input)
                y7 = self.softplus(x)
                ##relu6
                x = self.linear(input)
                y8 = self.ReLU6(x)
                ##mish
                x = self.linear(input)
                y9 = self.mish(x)
                ##Softmax
                x = self.linear(input)
                y10 = self.Softmax(x)
                ##concat
                y = torch.cat((y0, y1, y2, y4, y5, y6, y7, y8, y9, y10), 0)
                ##Softmax_2d
                x = self.linear(input)
                x = x.unsqueeze(dim=1)
                y2 = self.Softmax_2d(x)
                return y


        test_name = 'Activation'
        input_data = torch.randn(3, 100, 100).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Batch_Norm(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                ##BatchNorm1d need channel_size as 1, , greater than 1 will fail
                self.BatchNorm2d = nn.BatchNorm2d(1, affine=False)
                self.BatchNorm1d = nn.BatchNorm1d(1, affine=False)
                self.GroupNorm = nn.GroupNorm(3, 30)
                ##InstanceNorm2d and InstanceNorm1d need batch_size as 1 and w/h greater than 100 will sucess
                self.InstanceNorm2d = nn.InstanceNorm2d(1, affine=False)
                self.InstanceNorm1d = nn.InstanceNorm1d(1, affine=False)

            def forward(self, x):
                # x = self.BatchNorm2d(x)
                # x = torch.negative(x)
                # x = self.BatchNorm1d(x)
                # x = self.GroupNorm(x)
                x = self.InstanceNorm2d(x)
                # x = self.InstanceNorm1d(x)
                return x

        batch_size = 1
        test_name = 'Batch_Norm'

        input_data = torch.randn(batch_size, 1, 100, 100).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_ConvTranspose(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.dconv_2d = nn.ConvTranspose2d(40, 50, kernel_size=3, stride=2, padding=1, output_padding=0)
                self.dconv_1d = nn.ConvTranspose1d(50, 6, kernel_size=3, stride=2, padding=1, output_padding=0)

            def forward(self, x):
                ## output_shape = (input-1)*stride - 2*padding + kernel_size + output_padding
                x = self.dconv_2d(x) ## output shape (3, 50, 1, 59)
                x = torch.squeeze(x, 2) ## output shape (3, 50, 59)
                x = self.dconv_1d(x) ## output shape (3, 6, 118)
                return x

        batch_size = 3
        test_name = 'ConvTranspose'

        input_data = torch.randn(batch_size, 40, 1, 30).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_MaxPool(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                ## kersize = 3, stride = 2
                self.max_pool1d = nn.MaxPool1d(2, stride=2)
                self.max_pool2d = nn.MaxPool2d(2, stride=2, return_indices=True )
                self.max_unpool2d = nn.MaxUnpool2d(2, stride=2)

            def forward(self, x):
                ## output_shape = (h/w + kernel_size -1 - 1) / 2
                # output, indices = self.max_pool2d(x) ## output shape (3, 3, 50, 50)
                x = self.max_pool1d(x) ## output shape (3, 3, 50)
                # x = self.max_unpool2d(output, indices)
                return x

        batch_size = 3
        test_name = 'MaxPool'

        input_data = torch.randn(batch_size, 3, 100).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_MaxPool3d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2))

            def forward(self, x):
                x = self.max_pool3d(x)
                return x

        shape = [1, 4, 128, 20, 90]
        test_name = 'MaxPool3d'

        input_data = torch.randn(*shape).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_AvgPool(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.avgpool2d = nn.AvgPool2d(3, stride=2)

            def forward(self, x):
                ## output_shape = (hw - kernel_size) / stride + 1
                x = self.avgpool2d(x) ## output shape (3, 40, 14, 14)
                x = x.squeeze(dim=2)
                ## output_shape = (hw - kernel_size) / kernel_size + 1
                x = F.avg_pool1d(x, kernel_size=2)
                return x

        batch_size = 3
        test_name = 'AvgPool'

        input_data = torch.randn(batch_size, 40, 3, 30).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_ReflectionPad(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.ReflectionPad1d = nn.ReflectionPad1d(2)
                self.ReflectionPad2d = nn.ReflectionPad2d(2)

            def forward(self, x):
                x = self.ReflectionPad1d(x)
                # x = x.unsqueeze(dim=1)
                # x = self.ReflectionPad2d(x)
                return x

        batch_size = 3
        test_name = 'ReflectionPad'

        input_data = torch.randn(batch_size, 100, 100).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_ZeroPad2d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                ## left:3 right:4 up:4 down:6 postion pad zero
                self.ZeroPad2d = nn.ZeroPad2d(padding=(3, 4, 5, 6))

            def forward(self, x):
                ##input shape = (3, 100, 100)
                x = self.ZeroPad2d(x) ##output shape = (3, 111, 107)
                return x

        batch_size = 3
        test_name = 'ZeroPad2d'

        input_data = torch.randn(batch_size, 3, 100, 100).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Clip(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                return torch.clamp(x, min=-0.25, max=0.25)

        batch_size = 1
        test_name = 'Clip'

        input_data = torch.randn(batch_size, 100, 30).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)


    def test_ConstantPad(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.ConstantPad2d = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0)
                self.ConstantPad1d = nn.ConstantPad1d(2, 0.0)

            def forward(self, x):
                x = self.ConstantPad1d(x)
                x = x.unsqueeze(dim=1)
                x = self.ConstantPad2d(x)
                return x

        batch_size = 3
        test_name = 'ConstantPad'

        input_data = torch.randn(batch_size, 100, 30).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_Conv3d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv_test_3d = nn.Conv3d(3, 1, (3, 7, 7), stride=1, padding=0)

            def forward(self, x):
                ## 3d_conv for 5 dim
                x = self.conv_test_3d(x)
                return x

        input_shape = [1, 3, 30, 100, 10]
        test_name = 'Conv3d'
        ##torch needn't weight and bias
        input_data = torch.randn(input_shape)
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

    def test_MulConst(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.const_val = torch.randn(1)
                self.conv2d = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
            def forward(self, x):
                x = self.conv2d(x)
                x = torch.mul(x, self.const_val)
                x = torch.relu(x)
                return x

        input_shape = [1, 3, 100, 100]
        test_name = 'MulConst'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_name, torch_output_data)

if __name__ == "__main__":
    os.makedirs("torch_test", exist_ok=True)
    os.chdir("torch_test")
    tester = TORCH_IR_TESTER()
    if len(sys.argv) == 2:
        name = sys.argv[1]
        if name not in NOT_SUPPORT_INT8_TEST_IR:
            tester.set_quant_mode(mode="int8")
            tester.test_function.get(name)()
        if name not in NOT_SUPPORT_BF16_TEST_IR:
            tester.set_quant_mode(mode="bf16")
            tester.test_function.get(name)()
        exit(0)
    elif len(sys.argv) == 1:
        pass_list_i8 = list()
        pass_list_bf16 = list()

        for i in TEST_TORCH_IR:
            if i not in NOT_SUPPORT_INT8_TEST_IR:
                tester.test_function.get(i)()
                pass_list_i8.append(i)
                print("TEST {} Finish".format(i))

        for i in TEST_TORCH_IR:
            if i not in NOT_SUPPORT_BF16_TEST_IR:
                tester.set_quant_mode(mode="bf16")
                tester.test_function.get(i)()
                pass_list_bf16.append(i)
        print("Torch test result:")
        print("INT8 {} PASS {}".format("="*4, "="*4))
        for i in pass_list_i8:
            if i not in NOT_SUPPORT_INT8_TEST_IR:
                print("\t {}".format(i))

        print("BF16 {} PASS {}".format("="*4, "="*4))
        for i in pass_list_bf16:
            if i not in NOT_SUPPORT_BF16_TEST_IR:
                print("\t {}".format(i))

    else:
        print("Usage: test_torch.py ir_name")
        exit(-1)
