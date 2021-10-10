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
    "Conv2d", # Conv with 2d case
    "Conv1d", # Conv with 1d case
    "ConvTranspose1d",
    "Std",
    "Squeeze",
    "Linear",
    #"Mulit_attention_self", ## Low accuracy
    # "Mulit_attention_api",  ## now not support
    "Norm",
    "masked_fill",
    "Activation",
    "Cat_Chunk",
    "Math", ## sum, prod, log, min, max not support
    "Repeat",   ## repeat_interleave nonx not support
    # "Dropout", ## Dropout not support
    "LSTM",
    "GRU",
    "Size",
    "LayerNorm",
    "Mul_Add",
]

NOT_SUPPORT_CMDBUF_TEST_IR = [""]
NOT_SUPPORT_BF16_TEST_IR = []
NOT_SUPPORT_INT8_TEST_IR = [''] # just for save test time

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

class TORCH_IR_TESTER(object):
    def __init__(self):
        self.converter = None
        self.cvi_model_test = True

        self.test_function = {
            "LayerNorm": self.test_LayerNorm,
            "Conv2d": self.test_Conv2d,
            "Conv1d": self.test_Conv1d,
            "ConvTranspose1d": self.test_ConvTranspose1d,
            "Linear": self.test_Linear,
            "Conv2d": self.test_Conv2d,
            "Std": self.test_Std,
            "Squeeze": self.test_Squeeze,
            "Size": self.test_Size,
            "masked_fill": self.test_masked_fill,
            "Mulit_attention_self": self.test_Mulit_attention_self,
            "Mulit_attention_api": self.test_Mulit_attention_api,
            "Norm": self.test_Norm,
            "Activation": self.test_Activation,
            "Cat_Chunk": self.test_Cat_Chunk,
            "Math": self.test_Math,
            "Repeat": self.test_Repeat,
            "Dropout": self.test_Dropout,
            "LSTM": self.test_LSTM,
            "GRU": self.test_GRU,
            "Mul_Add": self.test_Mul_Add,
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
        if isinstance(input_data, dict):
            batch_size = input_data['input'].shape[0]
        else:
            batch_size = input_data.shape[0]
        converter = OnnxConverter(model_name, model_def, fp32_mlir, batch_size=batch_size)
        converter.run()
        del converter
        gc.collect()

        if isinstance(input_data, dict):
            for key, value in input_data.items():
                input_data[key] = value.data.numpy().astype(np.float32)
        else:
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
                                int8_csv, calib_table=table_name)
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
            ret = mlir_to_cvimodel(quant_mlir, cvimodel)
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
                            "--tolerance", "0.99,0.99,0.9"])
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
                            bf16_csv, all_bf16=True)
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
            ret = mlir_to_cvimodel(quant_mlir, cvimodel)
            if ret < 0: raise RuntimeError("gen_cvimodel failed")

            # run cvi_model
            output_tensor_npz = "{}_all_tensor_bf16_cvi.npz".format(model_name)
            cvi_outs = cvimodel_inference(bf16_tensors, cvimodel)
            assert(len(cvi_outs) == num_outputs)
            for name in cvi_outs:
                if name not in bf16_tensors:
                    raise RuntimeError("cvimodel output name not correct")
            np.savez(output_tensor_npz, **cvi_outs)
            npz_compare([output_tensor_npz, ref_npz, "--op_info", bf16_csv, "--tolerance", "0.9,0.9,0.9", "-vv"])

        del self.mlir_model


    def pytorch_transform_onnx(self, model, input_data, test_onnx_name, dynamic_axes_confirm=False):
        # Create some sample  input in the shape this model expects
        output_names = ['output']
        onnx_name = test_onnx_name+'.onnx'
        dynamic_axes_attr = {'input'  : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}} if dynamic_axes_confirm else None

        if type(input_data) == dict:
            input_names = list(input_data.keys())
            input_data = tuple(input_data.values())
        else:
            input_names = ['input']

        torch.onnx.export(model,
            input_data,
            onnx_name,
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_attr)

    def test_LSTM(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.rnn = nn.LSTM(
                    input_size=6,
                    hidden_size=5,
                    num_layers=1,
                    batch_first=True,
                )
                self.embedding_layer = torch.nn.Embedding(20, 6)

            def forward(self, x):
                x = self.embedding_layer(x)
                x = x.transpose_(1,0)
                r_out, (h_n, h_c) = self.rnn(x)
                return r_out

        test_onnx_name = 'LSTM'
        batch_size = 3
        seq_length = 4
        vocab_size=20
        input_data = np.random.uniform(0, 19, size=(batch_size, seq_length))
        input_data = torch.from_numpy(input_data).long()

        net = Net()
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Mul_Add(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x, y):
                out = torch.add(x, y)
                return out

        input_data = {}
        input_shape = [4, 1]
        input_data_temp = torch.randn(input_shape[0], input_shape[1])
        input_data['input'] = input_data_temp
        input_data['input1'] = input_data_temp
        test_onnx_name = 'Mul_Add'

        net = Net()
        torch_output_data = net(input_data['input'], input_data['input1'])

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        # torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_GRU(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.gru = nn.GRU(input_size=5, hidden_size=50, batch_first=True)
                self.embedding_layer = torch.nn.Embedding(3, 5)

            def forward(self, x):
                x = self.embedding_layer(x) ##shape: [1, 3 ,5]
                out, hidden = self.gru(x)
                return out

        test_onnx_name = 'GRU'
        input_data = torch.LongTensor([[0, 1, 2]]) ##shape: [1, 3]

        net = Net()
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Repeat(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = x.repeat(3, 4)
                # x = x.repeat_interleave(4, 0)
                x = torch.add(x, x)
                return x

        input_shape = [4, 1]
        test_onnx_name = 'Repeat'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

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

        input_shape = [3, 2, 8, 1]
        test_onnx_name = 'Squeeze'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Dropout(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.dropout = nn.Dropout(p=0.5)

            def forward(self, x):
                x = torch.negative(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.dropout(x)
                x = torch.add(x, x)
                return x

        input_shape = [4, 5]
        test_onnx_name = 'Repeat'

        net = Net()
        input_data = torch.randn(input_shape[0], input_shape[1])
        # normal = Normal(input_data, 5)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

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

        input_shape = [1, 1, 2, 3]
        test_onnx_name = 'Math'

        net = Net()
        input_data = torch.zeros(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

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

        input_shape = [1, 3, 20, 30]
        test_onnx_name = 'Cat_Chunk'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Size(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self,x):
                y = torch.ones(x.size(1))
                return torch.add(x, y)

        input_shape = [10, 16]
        test_onnx_name = "Size"

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_masked_fill(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.mask = torch.ByteTensor([[[1],[1],[0]],[[0],[1],[1]]])

            def forward(self,x):
                y = x.masked_fill(self.mask, value=torch.tensor(1.0))
                return y

        input_shape = [2,3,4]
        test_onnx_name = "masked_fill"

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)
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
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

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

        test_onnx_name = 'Mulit_attention_api'
        input_data = torch.LongTensor([[0, 1, 2]]) ##shape: [1, 3]

        net = Net()
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Mulit_attention_self(self):
        class SelfAttention(nn.Module):
            def __init__(self, hid_dim, n_heads, dropout, device):
                super().__init__()

                self.hid_dim = hid_dim
                self.n_heads = n_heads

                assert hid_dim % n_heads == 0

                self.w_q = nn.Linear(hid_dim, hid_dim)
                self.w_k = nn.Linear(hid_dim, hid_dim)
                self.w_v = nn.Linear(hid_dim, hid_dim)

                self.fc = nn.Linear(hid_dim, hid_dim)
                self.do = nn.Dropout(dropout)
                self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

            def forward(self, query, key, value, mask=None):
                bsz = query.shape[0]

                Q = self.w_q(query)
                K = self.w_k(key)
                V = self.w_v(value)

                Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                        self.n_heads).permute(0, 2, 1, 3)
                K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                        self.n_heads).permute(0, 2, 1, 3)
                V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                        self.n_heads).permute(0, 2, 1, 3)

                energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
                if mask is not None:
                    energy = energy.masked_fill(mask == 0, -1e10)
                attention = self.do(torch.softmax(energy, dim=-1))
                x = torch.matmul(attention, V)
                x = x.permute(0, 2, 1, 3).contiguous()
                x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
                x = self.fc(x)
                return x

        test_onnx_name = 'Mulit_attention_self'
        embed_dim = 2
        num_heads = 2
        batch_size = 3
        input_shape = [num_heads, batch_size, embed_dim]
        query = torch.randn(input_shape[0], input_shape[1], input_shape[2])
        key = torch.randn(input_shape[0], input_shape[1], input_shape[2])
        value = torch.randn(input_shape[0], input_shape[1], input_shape[2])
        input_data = {}
        input_data['input'] = query
        input_data['input1'] = key
        input_data['input2'] = value

        dropout = 0.2  # the dropout value
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        multihead_net = SelfAttention(embed_dim, num_heads, dropout, device)
        torch_output_data = multihead_net(input_data['input'], input_data['input1'], input_data['input2'])

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(multihead_net, input_data, test_onnx_name, False)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Norm(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = torch.negative(x)
                x = torch.norm(x, p=2, dim=1, keepdim=True)
                return x

        input_shape = [3, 1, 8, 1]
        test_onnx_name = 'Norm'

        net = Net()
        input_data = torch.randn(input_shape)
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
                x = torch.negative(x)
                x = torch.transpose(x, 1, 2)
                x = torch.squeeze(x)
                x = torch.add(x,x)
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

    def test_LayerNorm(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layer_norm = nn.LayerNorm([24, 50])

            def forward(self, x):
                # normal = Normal(x, 5)
                x = self.layer_norm(x)
                return x

        input_shape = [3, 24, 50]
        test_onnx_name = 'LayerNorm'

        net = Net()
        input_data = torch.randn(input_shape)
        torch_output_data = net(input_data)

        # Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name)

        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Activation(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.linear = nn.Linear(10, 20, bias=False)
                self.linear_return = nn.Linear(20, 10, bias=False)
                self.softplus = nn.Softplus()
                self.hardsigmoid = nn.Hardsigmoid()
                self.prelu = nn.PReLU()

            def forward(self, x):
                #tanh
                x = self.linear(x)
                x = torch.tanh(x)
                ##sigmoid
                x = self.linear_return(x)
                x = torch.sigmoid(x)
                ##relu
                x = self.linear(x)
                x = torch.relu(x)
                ##leaky_relu
                x = self.linear_return(x)
                x = F.leaky_relu(x)
                ##elu
                x = self.linear(x)
                x = F.elu(x)
                ##softplus
                x = self.linear_return(x)
                x = self.prelu(x)
                ##hardsigmoid
                x = self.linear(x)
                x = self.hardsigmoid(x)
                ##prelu
                x = self.linear_return(x)
                x = self.softplus(x)
                return x

        test_onnx_name = 'Activation'
        input_data = torch.randn(3, 6, 10).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_ConvTranspose1d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.dconv1 = nn.ConvTranspose1d(4, 6, kernel_size=3, stride=2, padding=1, output_padding=1)

            def forward(self, x):
                x = torch.negative(x)
                x = torch.squeeze(x, 3)
                x = self.dconv1(x)
                return x

        batch_size = 3
        test_onnx_name = 'ConvTranspose1d'

        input_data = torch.randn(batch_size, 4, 2, 1).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
        self.pytorch_transform_onnx(net, input_data, test_onnx_name, False)
        torch_output_data = torch_output_data.data.numpy()
        self.onnx_convert_and_infernece(input_data, test_onnx_name, torch_output_data)

    def test_Conv1d(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv_test = nn.Conv1d(in_channels=3,
                            out_channels=2,
                            kernel_size=3,
                            stride=2,
                            padding=1)

            def forward(self, x):
                x = torch.negative(x) ##tensor size [3, 3, 5]
                x = self.conv_test(x) ##tensor size [3, 2, 5]
                x = F.avg_pool1d(x, kernel_size=2)
                return x

        batch_size = 3
        test_onnx_name = 'Conv1d'
        input_data = torch.randn(batch_size, 3, 5).float()
        net = Net()
        torch_output_data = net(input_data)

        #Use the exporter from  torch to convert to onnx
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

        input_shape = [1, 3, 27, 27]
        test_onnx_name = 'Conv2d'
        ##torch needn't weight and bias
        input_data = torch.randn(input_shape)
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
    if len(sys.argv) == 2:
        tester.test_function.get(sys.argv[1])()
        tester.set_quant_mode(mode="bf16")
        tester.test_function.get(sys.argv[1])()
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


