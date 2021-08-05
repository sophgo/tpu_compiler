#!/usr/bin/env python3

from cvi_toolkit.model.mlir_model import MLIRModel
from cvi_toolkit.utils.mlir_shell import mlir_quant, \
     mlir_opt, mlir_to_cvimodel, run_cvimodel
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.numpy_helper import npz_compare
from cvi_toolkit.numpy_helper.npz_compare import fp32_to_bf16
import onnx
from onnx import helper
from onnx import TensorProto
import yaml
import onnxruntime
import pyruntime
import numpy as np
import os
import sys
import gc
import re

script_path = os.path.dirname(os.path.abspath(__file__))

TEST_ONNX_IR = [
    "Abs",
    "Add",
    "AddConst",
    "AveragePool",
#    "Concat",
    "Conv2d", # Conv with 2d case
    "Conv4Bit", # Conv, filter will quant to 4bit
    # "Conv3d", # Conv with 3d case
    "DepthToSpace",
    "FullyConnected",
    "GroupFC", # test Group FC
    "Gather",
    "GlobalMaxPool",
    "GRU",
    "GRUh", # test gru output Y_h
    "LeakyRelu",
    "LRN",
    "LSTM",
    "Max",
    "Min",
    "Mul",
    "MatMul",
    "Neg",
    "Pad",
    "Relu",
    "PRelu",
#    "ReduceMax",
    "ReduceMean",
    "ResizeNearest",
    "ResizeLinear",
    "ResizePytorch",
#   "ResizeModel",
#    "Reciprocal",
    "Slice",
    "Slice_3dim", # test slice for 3 dims
    "Sigmoid",
    "Sub",
    "Sum",
    "Softmax",
    "Tile",
    "Upsample",
#    "Transpose",
]

NOT_SUPPORT_CMDBUF_TEST_IR = ["DepthToSpace"]
NOT_SUPPORT_BF16_TEST_IR = ["Relu", "LRN", "Max", "Min", "PRelu", "Reciprocal", "Conv4Bit", "Transpose", "Sum"]
NOT_SUPPORT_INT8_TEST_IR = ["Gather", "Softmax"] # just for save test time

QUANT_BITWIDTH = {}

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
        for key,value in QUANT_BITWIDTH.items():
            f.write("bitwidth {} {}\n".format(key, value))

def _onnx_inference(input, model_name, input_name="input", input_cb=None):
    ort_session = onnxruntime.InferenceSession(model_name)
    if callable(input_cb):
        ort_inputs = input_cb(model_name, "onnx", input)
    else:
        ort_inputs = {input_name: input}
    outs = ort_session.run(None, ort_inputs)
    ort_outputs = ort_session.get_outputs()
    outputs = {}
    idx = 0
    for output in ort_outputs:
        outputs[output.name] = outs[idx]
        idx = idx + 1
    return outputs

def onnx_inference(input, model_def, model_name, input_cb = None):
    model = "{}.onnx".format(model_name)
    onnx.save(model_def, model)
    return _onnx_inference(input, model, input_cb=input_cb)

def cvimodel_inference(input, model_name):
    model = pyruntime.Model(model_name)
    assert(model != None)
    model_in = model.inputs[0].data
    assert(model_in.dtype == input.dtype)
    assert(model_in.size == input.size)
    model_in[:] = input.reshape(model_in.shape)
    model.forward()
    outputs = {}
    for output in model.outputs:
        outputs[output.name] = np.array(output.data)
    return outputs

class ONNX_IR_TESTER(object):
    def __init__(self):
        self.converter = None
        self.cvi_model_test = True

        self.test_function = {
            "Abs": self.test_Abs,
            "Add": self.test_Add,
            "AddConst": self.test_AddConst,
            "AveragePool": self.test_AveragePool,
            "Concat": self.test_Concat,
            "Conv2d": self.test_Conv2d,
            "Conv4Bit": self.test_Conv4Bit,
            "Conv3d": self.test_Conv3d,
            "DepthToSpace": self.test_DepthToSpace,
            "FullyConnected": self.test_FullyConnected,
            "GroupFC": self.test_GroupFC,
            "Gather": self.test_Gather,
            "GlobalMaxPool": self.test_GlobalMaxPool,
            "GRU": self.test_GRU,
            "GRUh": self.test_GRUh,
            "LeakyRelu": self.test_LeakyRelu,
            "LRN": self.test_LRN,
            "LSTM": self.test_LSTM,
            "Max": self.test_Max,
            "Min": self.test_Min,
            "Mul": self.test_Mul,
            "MatMul": self.test_MatMul,
            "Neg": self.test_Neg,
            "PRelu": self.test_PRelu,
            "Reciprocal": self.test_Reciprocal,
            "Pad": self.test_Pad,
            "Relu": self.test_Relu,
            "ResizeNearest": self.test_ResizeNearest,
            "ResizeLinear": self.test_ResizeLinear,
            "ResizePytorch": self.test_ResizePytorch,
            "ResizeModel": self.test_ResizeModel,
            "Slice": self.test_Slice,
            "Slice_3dim": self.test_Slice_3dim,
            "Sigmoid": self.test_Sigmoid,
            "Sub": self.test_Sub,
            "Sum": self.test_Sum,
            "Softmax": self.test_Softmax,
            "Transpose": self.test_Transpose,
            "Tile": self.test_Tile,
            "ReduceMean": self.test_ReduceMean,
            "ReduceMax": self.test_ReduceMax,
            "Upsample": self.test_Upsample,
        }
        self.set_quant_mode()

    def set_quant_mode(self, mode="int8"):
        if mode == "int8":
            self.quant_mode = "int8"
        elif mode == "bf16":
            self.quant_mode = "bf16"
        else:
            raise RuntimeError("Not support quant mode {}".format(mode))

    def onnx_convert_and_infernece(self, input_data, model_def, model_name, input_cb=None):
        fp32_mlir = "{}.mlir".format(model_name)
        self.converter = OnnxConverter(model_name, model_def, fp32_mlir)
        self.converter.run()
        del self.converter
        gc.collect()

        onnx_outs = onnx_inference(input_data, model_def, model_name, input_cb)
        input_data = input_data.astype(np.float32)
        num_outputs = len(onnx_outs)
        input_npz = "{}_input_fp32.npz".format(model_name)
        np.savez(input_npz, input=input_data)
         # opt
        fp32_opt_mlir = "{}_opt.mlir".format(model_name)
        fp32_csv = "{}_fp32.csv".format(model_name)
        mlir_opt(fp32_mlir, fp32_opt_mlir, fp32_csv)
        self.mlir_model = None
        self.mlir_model = MLIRModel()
        self.mlir_model.load_model(fp32_opt_mlir)
        mlir_outs = self.mlir_model.inference(input_data)
        fp32_tensors = self.mlir_model.get_all_tensor()
        # Test output
        assert(len(mlir_outs) == num_outputs)
        if num_outputs > 1:
            patten = re.compile(r"_[A-Z]\w+?$")
            for name in mlir_outs:
                onnx_name = patten.sub("", name)
                print("Compare mlir[{}] : onnx[{}]".format(name, onnx_name))
                np.testing.assert_allclose(mlir_outs[name], onnx_outs[onnx_name], rtol=1e-5, atol=1e-01)
        else:
            mlir_out = list(mlir_outs.values())[0]
            onnx_out = onnx_outs.popitem()[1]
            np.testing.assert_allclose(mlir_out, onnx_out, rtol=1e-5, atol=1e-01)

        mlir_npz = "{}_fp32.npz".format(model_name)
        np.savez(mlir_npz, **fp32_tensors)

        if self.cvi_model_test:
            for i in NOT_SUPPORT_CMDBUF_TEST_IR:
                if i == model_name:
                    print("{} not support cmdbuf test!".format(model_name))
                    return

            tensors = self.mlir_model.get_all_tensor()
            if self.quant_mode == "int8":
                for i in NOT_SUPPORT_INT8_TEST_IR:
                    if i == model_name:
                        print("{} not support bf16 test!".format(model_name))
                        return
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
                             "0.6,0.6,0.6", "--dequant", "--op_info", int8_csv])

                # gen cvimodel
                cvimodel = "{}_int8.cvimodel".format(model_name)
                ret = mlir_to_cvimodel(quant_mlir, cvimodel)
                if ret < 0: raise RuntimeError("gen_cvimodel failed")

                # run cvi_model
                output_tensor_npz = "{}_all_tensor_int8_cvi.npz".format(model_name)
                input_i8 = None
                if 'input_quant_i8' in int8_tensors:
                    input_i8 = int8_tensors['input_quant_i8'].astype(np.int8)
                elif 'input_quant_u16' in int8_tensors:
                    input_i8 = int8_tensors['input_quant_u16'].astype(np.uint16)

                cvi_outs = cvimodel_inference(input_i8, cvimodel)
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
                input_bf16 = None
                if 'input_quant_i16' in bf16_tensors:
                    input_bf16 = tensors['input'].astype(np.int16)
                elif 'input_quant_u16' in bf16_tensors:
                    input_bf16 = tensors['input'].astype(np.uint16)
                else:
                    input_bf16 = tensors['input']
                cvi_outs = cvimodel_inference(input_bf16, cvimodel)
                assert(len(cvi_outs) == num_outputs)
                for name in cvi_outs:
                    if name not in bf16_tensors:
                        raise RuntimeError("cvimodel output name not correct")
                np.savez(output_tensor_npz, **cvi_outs)
                npz_compare([output_tensor_npz, ref_npz, "--op_info", bf16_csv, "--tolerance", "0.9,0.9,0.9", "-vv"])

        del self.mlir_model

    def test_model(self, input_shape, model_path, input_name="input"):
        if isinstance(input_shape, list):
            input_shape = [int(x) for x in input_shape]
            input_shape = tuple(input_shape)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        model_name = model_path.split("/")[-1].split(".")[0]
        onnx_model = onnx.load(model_path)

        fp32_mlir = "{}.mlir".format(model_name)

        self.converter = OnnxConverter(model_name, onnx_model, fp32_mlir)
        self.converter.run()
        del self.converter
        gc.collect()
        onnx_outs = _onnx_inference(input_data, model_path, input_name)

        self.mlir_model = MLIRModel()
        self.mlir_model.load_model(fp32_mlir)
        mlir_outs = self.mlir_model.inference(input_data)
        # Test output
        assert(len(mlir_outs) == len(onnx_outs))
        patten = re.compile(r"_\w+?$")
        for name in mlir_outs:
            onnx_name = patten.sub("", name)
            print("Compare mlir[{}] : onnx[{}]".format(name, onnx_name))
            np.testing.assert_allclose(mlir_outs[name], onnx_outs[onnx_name], rtol=1e-5, atol=1e-01)

        del self.mlir_model

        print("PASS")

    def test_Abs(self):
        test_case = 'Abs'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 6, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        #test three input
        concat_def = helper.make_node(
            'Concat',  # node name
            ['input', 'X1'],  # inputs
            ['X2'],  # outputs
            axis = 1
        )

        abs_def = helper.make_node(
            'Abs',
            ['X2'],
            ['output'],
        )

        graph_def = helper.make_graph(
            [x1_def, concat_def, abs_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_AddConst(self):
        test_case = 'AddConst'
        input_shape = [1, 3, 28, 28]
        output_shape = [1, 3, 28, 28]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)
        w_data = np.random.rand(input_shape[0], input_shape[1],
                                input_shape[2], input_shape[3]).astype(np.float32)
        w_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=w_data.shape,
                vals=w_data.flatten(),
            ),
        )

        add_node = helper.make_node(
            'Add',  # node name
            ['input', 'w'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph(
            [w_node_def, add_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Add(self):
        test_case = 'Add'
        input_shape = [1, 3, 8, 8]
        output_shape = [1, 3, 8, 8]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        reduce1_node = helper.make_node(
            'ReduceMax',
            ['input'],
            ['reduce1'],
            keepdims=1,
            axes=[1, ],
        )
        reduce2_node = helper.make_node(
            'ReduceMax',
            ['reduce1'],
            ['reduce2'],
            keepdims=1,
            axes=[3, ],
        )
        transpose_node = helper.make_node(
            "Transpose",
            ['reduce2'],
            ['trans'],
            perm=(0,1,3,2),
        )

        add_node = helper.make_node(
            'Add',  # node name
            ['input', 'trans'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph(
            [reduce1_node, reduce2_node, transpose_node, add_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_AveragePool(self):
        test_case = 'AveragePool'
        input_data = np.random.randn(1, 3, 28, 28).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30, 30])

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        node_def = onnx.helper.make_node(
            "AveragePool",
            inputs=['X1'],
            outputs=['output'],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[2, 2, 2, 2],
            count_include_pad=1
        )
        graph_def = helper.make_graph(
            [x1_def, node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Conv2d(self):
        test_case = 'Conv2d'
        batch_size = 4
        input_data = np.random.randn(batch_size, 3, 5, 5).astype(np.float32)
        weight_data = np.random.randn(3, 3, 3, 3).astype(np.float32)
        bias_data = np.random.randn(3).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [batch_size, 3, 5, 5])

        weight_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=weight_data.shape,
                vals=weight_data.flatten(),
            ),
        )
        bias_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_b'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=bias_data.shape,
                vals=bias_data.flatten(),
            ),
        )
        node_def = onnx.helper.make_node(
            "Conv",
            inputs=['input', 'conv_w', 'conv_b'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )
        graph_def = helper.make_graph(
            [weight_node_def, bias_node_def, node_def],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Conv4Bit(self):
        test_case = 'Conv4Bit'
        input_data = np.random.randn(4, 3, 5, 5).astype(np.float32)
        weight_data = np.random.randn(3, 3, 3, 3).astype(np.float32)
        bias_data = np.random.randn(3).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [4, 3, 5, 5])

        weight_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=weight_data.shape,
                vals=weight_data.flatten(),
            ),
        )
        QUANT_BITWIDTH['conv_w'] = 4
        bias_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_b'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=bias_data.shape,
                vals=bias_data.flatten(),
            ),
        )
        node_def = onnx.helper.make_node(
            "Conv",
            inputs=['input', 'conv_w', 'conv_b'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )
        graph_def = helper.make_graph(
            [weight_node_def, bias_node_def, node_def],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        self.onnx_convert_and_infernece(input_data, model_def, test_case)
        QUANT_BITWIDTH.clear()

    def test_Conv3d(self):
        test_case = 'Conv3d'
        input_data = np.random.randn(1, 16, 10, 50, 100).astype(np.float32)
        weight_data = np.random.randn(33, 16, 3, 5, 2).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, 33, 8, 50, 99])

        weight_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=weight_data.shape,
                vals=weight_data.flatten().astype(int),
            ),
        )
        node_def = onnx.helper.make_node(
            "Conv",
            inputs=['input', 'conv_w'],
            outputs=['output'],
            kernel_shape=[3, 5, 2],
            pads=[4, 2, 0, 4, 2, 0],
            strides=[2, 1, 1],
            dilations=[1, 1, 1],
            group=1,
        )
        graph_def = helper.make_graph(
            [weight_node_def, node_def],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_DepthToSpace(self):
        test_case = 'DepthToSpace'
        in_shape = [1, 640, 10, 8]
        in_shape = [1, 8, 3, 4]
        in_shape = [1, 32, 3, 4]
        in_shape = [1, 128, 3, 4]
        in_shape = [1, 256, 3, 4]
        in_shape = [1, 256, 4, 4]
        in_shape = [1, 256, 54, 96]
        in_shape = [1, 256, 540, 960]
        in_shape = [1, 256, 1080, 1920]
        n, c, h, w = in_shape
        blocksize = 2
        mode='CRD'
        mode='DCR' # default
        out_shape = [n, c // (blocksize*blocksize), h * blocksize, w * blocksize]
        input_data = np.arange(np.prod(in_shape)).reshape(in_shape).astype(np.float32)
        input_data = np.random.rand(*in_shape).reshape(in_shape).astype(np.float32)
        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, out_shape)
        node_def = onnx.helper.make_node(
            "DepthToSpace",
            mode=mode,
            blocksize=blocksize,
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph(
            [node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Gather(self):
        test_case = 'Gather'
        total_tokens = 60004
        input_shape = [1, 13]
        output_shape = [1, 13, 256]
        input_data = np.random.randint(0, total_tokens, input_shape).astype(np.int64)
        token_data = np.random.randn(total_tokens, 256).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.INT64, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        token_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['tokens'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=token_data.shape,
                vals=token_data.flatten(),
            ),
        )

        gather_node = helper.make_node(
            'Gather',  # node name
            ['tokens','input'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [token_def, gather_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_FullyConnected(self):
        test_case = 'FullyConnected'
        B = 4 # batch only for input
        M = 16
        K = 40
        N = 64

        input_data = np.random.rand(B, M, K).astype(np.float32)
        filter_data = np.random.rand(K, N).astype(np.float32)
        div_data = np.array([8.0])

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [B, M, N])

        filter_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['filter'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=filter_data.shape,
                vals=filter_data.flatten(),
            ),
        )

        div_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['div'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=div_data.shape,
                vals=div_data.flatten(),
            ),
        )

        fc_node = helper.make_node(
            'MatMul',  # node name
            ['input', 'filter'],  # inputs
            ['fc'],  # outputs
        )

        scale_node = helper.make_node(
            "Div",
            ['fc', 'div'],
            ['output'],
        )

        graph_def = helper.make_graph(
            [filter_def, div_def, fc_node, scale_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_GroupFC(self):
        test_case = 'GroupFC'
        input_shape = [16, 40, 43]
        filter_shape = [16, 43, 48]
        bias_shape = [16, 1, 48]
        output_shape = [16, 40, 48]

        input_data = np.random.rand(np.prod(input_shape)).reshape(
            input_shape).astype(np.float32)
        filter_data = np.random.rand(np.prod(filter_shape)).reshape(
            filter_shape).astype(np.float32)
        bias_data = np.random.rand(np.prod(bias_shape)).reshape(
            bias_shape).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        filter_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['filter'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=filter_data.shape,
                vals=filter_data.flatten(),
            ),
        )
        bias_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['bias'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=bias_data.shape,
                vals=bias_data.flatten(),
            ),
        )

        fc_node = helper.make_node(
            'MatMul',  # node name
            ['input', 'filter'],  # inputs
            ['fc'],  # outputs
        )
        add_node = helper.make_node(
            'Add',
            ['fc', 'bias'],
            ['output'],
        )

        graph_def = helper.make_graph(
            [filter_def, bias_def, fc_node, add_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_GlobalMaxPool(self):
        test_case = 'GlobalMaxPool'
        input_data = np.random.randn(1, 3, 28, 28).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 1])
        node_def = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=['input'],
            outputs=['output'],
        )
        graph_def = helper.make_graph(
            [node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_GRU(self):
        test_case = 'GRU'
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 64
        hidden_size = 32
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(
            seq_length, batch_size, input_size).astype(np.float32)
        h_data = np.random.rand(num_dir, batch_size,
                                hidden_size).astype(np.float32)
        w_data = np.random.rand(
            num_dir, 3*hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(
            num_dir, 3*hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6*hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [seq_length, num_dir, batch_size, hidden_size])

        w_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=w_data.shape,
                vals=w_data.flatten(),
            ),
        )
        r_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['r'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=r_data.shape,
                vals=r_data.flatten(),
            ),
        )
        b_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['b'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=b_data.shape,
                vals=b_data.flatten(),
            ),
        )
        h_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['h'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=h_data.shape,
                vals=h_data.flatten(),
            ),
        )
        node_def = onnx.helper.make_node(
            "GRU",
            inputs=['input', 'w', 'r', 'b', '', 'h'],
            outputs=['output', ''],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph(
            [w_node_def, r_node_def, b_node_def, h_node_def, node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_GRUh(self):
        test_case = 'GRUh'
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(
            seq_length, batch_size, input_size).astype(np.float32)
        h_data = np.random.rand(num_dir, batch_size,
                                hidden_size).astype(np.float32)
        w_data = np.random.rand(
            num_dir, 3*hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(
            num_dir, 3*hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 6*hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [num_dir, batch_size, hidden_size])

        w_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=w_data.shape,
                vals=w_data.flatten(),
            ),
        )
        r_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['r'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=r_data.shape,
                vals=r_data.flatten(),
            ),
        )
        b_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['b'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=b_data.shape,
                vals=b_data.flatten(),
            ),
        )
        h_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['h'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=h_data.shape,
                vals=h_data.flatten(),
            ),
        )
        node_def = onnx.helper.make_node(
            "GRU",
            inputs=['input', 'w', 'r', 'b', '', 'h'],
            outputs=['', 'output'],
            direction=direction,
            hidden_size=hidden_size,
            linear_before_reset=1,
        )
        graph_def = helper.make_graph(
            [w_node_def, r_node_def, b_node_def, h_node_def, node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_LeakyRelu(self):
        alpha = 0.01
        test_case = "LeakyRelu"
        input_shape = [1, 3, 224, 224]
        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )
        node_def = helper.make_node(
            "LeakyRelu", # node name
            ['X1'], # inputs
            ['output'], # outputs
            alpha=alpha
        )

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [x1_def, node_def],
            test_case,
            [input],
            [output],
        )

        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_LRN(self):
        test_case = 'LRN'
        input_shape = [1, 10, 27, 27]
        output_shape = [1, 10, 27, 27]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        lrn_def = helper.make_node(
            'LRN',  # node name
            ['X1'],  # inputs
            ['output'],  # outputs
            size=5,
        )

        graph_def = helper.make_graph(
            [x1_def, lrn_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        #only support positive input for lrn
        input_data = -input_data

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_LSTM(self):
        test_case = 'LSTM'
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(
            seq_length, batch_size, input_size).astype(np.float32)
        w_data = np.random.rand(
            num_dir, 4*hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(
            num_dir, 4*hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 8*hidden_size).astype(np.float32)
        h0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        c0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [seq_length, num_dir, batch_size, hidden_size])

        w_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=w_data.shape,
                vals=w_data.flatten(),
            ),
        )
        r_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['r'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=r_data.shape,
                vals=r_data.flatten(),
            ),
        )
        b_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['b'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=b_data.shape,
                vals=b_data.flatten(),
            ),
        )
        h0_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['h0'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=h0_data.shape,
                vals=h0_data.flatten(),
            ),
        )
        c0_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['c0'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=c0_data.shape,
                vals=c0_data.flatten(),
            ),
        )
        node_def = onnx.helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0','c0'],
            outputs=['output','',''],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph(
            [w_node_def, r_node_def, b_node_def, h0_node_def, c0_node_def, node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Max(self):
        test_case = 'Max'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        #test only one input
        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        #test three input
        max_def = helper.make_node(
            'Max',  # node name
            ['input', 'X1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [x1_def, max_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Min(self):
        test_case = 'Min'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        #test only one input

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        #test four input
        min_def = helper.make_node(
            'Min',  # node name
            ['input', 'X1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [x1_def, min_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Mul(self):
        # mul(1x16x28x28, 1x1x28x28) => 1x16x28x28
        test_case = 'BroadcastMul'
        input_shape = [1, 16, 28, 28]
        output_shape = [1, 16, 28, 28]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        reduce_node = helper.make_node(
            'ReduceMax',
            ['input'],
            ['X1'],
            keepdims=1,
            axes=[1, ],
        )

        neg_node = helper.make_node(
            'Neg',  # node name
            ['X1'],  # inputs
            ['X2'],  # outputs
        )

        #test only one input
        mul_node = helper.make_node(
            'Mul',  # node name
            ['input', "X2"],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [reduce_node, neg_node, mul_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_MatMul(self):
        # matmul(1x16x40x64, 1x16x64x40) => 1x16x40x40
        test_case = 'MatMul'
        input_shape = [1, 16, 40, 64]
        output_shape = [1, 16, 40, 40]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_node = helper.make_node(
            'Transpose',
            ['input'],
            ['X1'],
            perm = (0,1,3,2),
        )

        matmul_node = helper.make_node(
            'MatMul',  # node name
            ['input','X1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [x1_node, matmul_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Neg(self):
        test_case = 'Neg'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        neg_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph(
            [neg_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_PRelu(self):
        test_case = 'PRelu'
        input_shape = [1, 3, 27, 27]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        weight_data = np.random.rand(input_shape[0], input_shape[1],
                           1, 1).astype(np.float32)
        prelu_weight_constant = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['prelu_weight'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=weight_data.shape,
                vals=weight_data.flatten().astype(int),
            ),
        )

        output_shape = [1, 3, 27, 27]
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)
        prelu_def = helper.make_node(
            'PRelu',  # node name
            ['input', 'prelu_weight'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [prelu_weight_constant, prelu_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        model_name = '{}.onnx'.format(test_case)
        onnx.save(model_def, model_name)
        input_data = np.random.randn(input_shape[0], input_shape[1],
                            input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Reciprocal(self):
        test_case = 'Reciprocal'
        input_shape = [1, 3, 224, 224]
        node_def = helper.make_node(
            "Reciprocal", # node name
            ['input'], # inputs
            ['X1'], # outputs
        )

        neg_def = helper.make_node(
            "Neg", # node name
            ['X1'], # inputs
            ['output'], # outputs
        )

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def, neg_def],
            test_case,
            [input],
            [output],
        )

        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        # avoid divide 0
        input_data[input_data==0] = 1
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Pad(self):
        test_case = 'Pad'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 6, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        pads = np.array([0, 0, 1, 4, 0, 0, 2, 3]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        #pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        for idx, p in enumerate(pads):
            dim = idx % 4
            output_shape[dim] = int(output_shape[dim] + p)

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        #test three input
        concat_def = helper.make_node(
            'Concat',  # node name
            ['input', 'X1'],  # inputs
            ['X2'],  # outputs
            axis = 1
        )

        pad_def  = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['pads'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=pads.shape,
                vals=pads.flatten(),
            ),
        )

        relu_def = helper.make_node(
            'Pad',
            ['X2', 'pads'],
            ['output'],
            mode='edge'
        )

        graph_def = helper.make_graph(
            [x1_def, concat_def, pad_def, relu_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Relu(self):
        test_case = 'Relu'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 6, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        #test three input
        concat_def = helper.make_node(
            'Concat',  # node name
            ['input', 'X1'],  # inputs
            ['X2'],  # outputs
            axis = 1
        )

        relu_def = helper.make_node(
            'Relu',
            ['X2'],
            ['output'],
        )

        graph_def = helper.make_graph(
            [x1_def, concat_def, relu_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_ReduceMean(self):
        test_case = "ReduceMean"
        input_shape = [1, 3, 4, 128]
        output_shape = [1, 3, 4, 1]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )

        reduce_node = helper.make_node(
            'ReduceMean',
            ['X1'],
            ['output'],
            keepdims=1,
            axes=[3, ],
        )

        graph_def = helper.make_graph(
            [x1_node, reduce_node],
            test_case,
            [input],
            [output]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_ReduceMax(self):
        test_case = "ReduceMax"
        input_shape = [1, 128, 4, 4]
        output_shape = [1, 1, 4, 4]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )

        reduce_node = helper.make_node(
            'ReduceMax',
            ['X1'],
            ['output'],
            keepdims=1,
            axes=[1, ],
        )

        graph_def = helper.make_graph(
            [x1_node, reduce_node],
            test_case,
            [input],
            [output]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        input_data = input_data.reshape(tuple(input_shape))
        indices = np.argmax(input_data, axis=1)
        input_data[:, indices] += 0.1
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_ResizeNearest(self):
        test_case = "test_ResizeNearest"
        input_shape = [1, 96, 3, 4]
        output_shape = [1, 96, 6, 8]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)


        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)
        sizes = np.array([1, 96, 6, 8], dtype=np.int64)

        roi_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['roi'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=roi.shape,
                vals=roi.flatten(),
            ),
        )
        scales_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scales'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=scales.shape,
                vals=scales.flatten(),
            ),
        )
        sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=sizes.shape,
                vals=sizes.flatten(),
            ),
        )
        resize_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes'],
            outputs=['output'],
            mode='nearest',
        )

        graph_def = helper.make_graph(
            [roi_node_def, scales_node_def, sizes_node_def, resize_node],
            test_case,
            [input],
            [output]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_ResizeLinear(self):
        # by cpu, with two outputs, scale is not integer
        test_case = "test_ResizeLinear"
        input_shape = [1, 32, 208, 30]
        output_shape = [1, 2, 416, 48]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)
        X2 = helper.make_tensor_value_info(
            'X2', TensorProto.FLOAT, [1, 32, 416, 48])

        sizes = np.array([1, 32, 416, 48], dtype=np.int64)
        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)
        roi_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['roi'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=roi.shape,
                vals=roi.flatten(),
            ),
        )
        scales_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scales'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=scales.shape,
                vals=scales.flatten(),
            ),
        )
        sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=sizes.shape,
                vals=sizes.flatten(),
            ),
        )
        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )
        resize_node = helper.make_node(
            'Resize',
            inputs=['X1', 'roi', 'scales', 'sizes'],
            outputs=['X2'],
            mode='linear',
            coordinate_transformation_mode='half_pixel'
        )
        weight_data = np.random.randn(2, 32, 3, 3).astype(np.float32)
        bias_data = np.random.randn(2).astype(np.float32)

        weight_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_w'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=weight_data.shape,
                vals=weight_data.flatten(),
            ),
        )
        bias_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['conv_b'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=bias_data.shape,
                vals=bias_data.flatten(),
            ),
        )
        conv_node_def = onnx.helper.make_node(
            "Conv",
            inputs=['X2', 'conv_w', 'conv_b'],
            outputs=['output'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )

        graph_def = helper.make_graph(
            [x1_node, roi_node_def, scales_node_def, sizes_node_def, resize_node, weight_node_def, bias_node_def, conv_node_def],
            test_case,
            [input],
            [output, X2]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_ResizePytorch(self):
        # by npu, scale is integer
        test_case = "test_ResizePytorch"
        input_shape = [1, 32, 24, 12]
        output_shape = [1, 32, 48, 96]
        #input_shape = [1, 32, 6, 48]
        #output_shape = [1, 32, 12, 96]
        #input_shape = [1, 2, 6, 8]
        #output_shape = [1, 2, 12, 16]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)
        sizes = np.array(output_shape, dtype=np.int64)

        roi_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['roi'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=roi.shape,
                vals=roi.flatten(),
            ),
        )
        scales_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scales'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=scales.shape,
                vals=scales.flatten(),
            ),
        )
        sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=sizes.shape,
                vals=sizes.flatten(),
            ),
        )
        x1_node = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )
        resize_node = helper.make_node(
            'Resize',
            inputs=['X1', 'roi', 'scales', 'sizes'],
            outputs=['output'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
            #coordinate_transformation_mode='align_corners'
        )

        graph_def = helper.make_graph(
            [x1_node, roi_node_def, scales_node_def, sizes_node_def, resize_node],
            test_case,
            [input],
            [output]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        #input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
        #input_data = np.arange(
        #        0, 1, 1 / np.prod(input_shape)).flatten()[:np.prod(input_shape)].reshape(input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)



    def test_ResizeModel(self):
        test_case = "test_ResizeModel"
        input_shape = [1, 32, 3, 24]
        x1_shape = [1, 32, 6, 48]
        x2_shape = [1, 32, 12, 96]
        x3_shape = [1, 32, 24, 192]
        x4_shape = [1, 32, 48, 384]
        #input_shape = [1, 32, 6, 48]
        #output_shape = [1, 32, 12, 96]
        #input_shape = [1, 2, 6, 8]
        #output_shape = [1, 2, 12, 16]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        x3_4 = helper.make_tensor_value_info(
            'X3_4', TensorProto.FLOAT, x3_shape)
        x4 = helper.make_tensor_value_info(
            'X4', TensorProto.FLOAT, x4_shape)
        x32 = helper.make_tensor_value_info(
            'X32', TensorProto.FLOAT, x3_shape)
        x3 = helper.make_tensor_value_info(
            'X3', TensorProto.FLOAT, x3_shape)
        x2_4 = helper.make_tensor_value_info(
            'X2_4', TensorProto.FLOAT, x2_shape)
        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)
        x1_sizes = np.array(x1_shape, dtype=np.int64)
        x2_sizes = np.array(x2_shape, dtype=np.int64)
        x3_sizes = np.array(x3_shape, dtype=np.int64)
        x4_sizes = np.array(x4_shape, dtype=np.int64)

        roi_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['roi'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=roi.shape,
                vals=roi.flatten(),
            ),
        )
        scales_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scales'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=scales.shape,
                vals=scales.flatten(),
            ),
        )
        x1_sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['x1_sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=x1_sizes.shape,
                vals=x1_sizes.flatten(),
            ),
        )
        x2_sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['x2_sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=x2_sizes.shape,
                vals=x2_sizes.flatten(),
            ),
        )
        x3_sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['x3_sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=x3_sizes.shape,
                vals=x3_sizes.flatten(),
            ),
        )
        x4_sizes_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['x4_sizes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=x4_sizes.shape,
                vals=x4_sizes.flatten(),
            ),
        )
        neg_node = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['input_neg'],  # outputs
        )
        x1_resize_node = helper.make_node(
            'Resize',
            inputs=['input_neg', 'roi', 'scales', 'x1_sizes'],
            outputs=['X1'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x12_resize_node = helper.make_node(
            'Resize',
            inputs=['input_neg', 'roi', 'scales', 'x1_sizes'],
            outputs=['X12'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x2_resize_node = helper.make_node(
            'Resize',
            inputs=['X1', 'roi', 'scales', 'x2_sizes'],
            outputs=['X2'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x22_resize_node = helper.make_node(
            'Resize',
            inputs=['X12', 'roi', 'scales', 'x2_sizes'],
            outputs=['X22'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x3_resize_node = helper.make_node(
            'Resize',
            inputs=['X2', 'roi', 'scales', 'x3_sizes'],
            outputs=['X3'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x32_resize_node = helper.make_node(
            'Resize',
            inputs=['X22', 'roi', 'scales', 'x3_sizes'],
            outputs=['X32'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x33_resize_node = helper.make_node(
            'Resize',
            inputs=['X22', 'roi', 'scales', 'x3_sizes'],
            outputs=['X33'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x34_resize_node = helper.make_node(
            'Resize',
            inputs=['X33', 'roi', 'scales', 'x3_sizes'],
            outputs=['X34'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x4_resize_node = helper.make_node(
            'Resize',
            inputs=['X34', 'roi', 'scales', 'x4_sizes'],
            outputs=['X4'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x2_4_resize_node = helper.make_node(
            'Resize',
            inputs=['input_neg', 'roi', 'scales', 'x2_sizes'],
            outputs=['X2_4'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        x3_4_resize_node = helper.make_node(
            'Resize',
            inputs=['X1', 'roi', 'scales', 'x3_sizes'],
            outputs=['X3_4'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        graph_def = helper.make_graph(
            [neg_node, roi_node_def, scales_node_def, x1_sizes_node_def, x2_sizes_node_def, x3_sizes_node_def, x4_sizes_node_def, x1_resize_node, x12_resize_node,
                x2_resize_node, x22_resize_node, x3_resize_node, x32_resize_node, x33_resize_node, x34_resize_node, x4_resize_node, x2_4_resize_node, x3_4_resize_node],
            test_case,
            [input],
            [x3_4, x2_4, x4, x32, x3]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        #input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
        #input_data = np.arange(
        #        0, 1, 1 / np.prod(input_shape)).flatten()[:np.prod(input_shape)].reshape(input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)


    def test_Slice(self):
        _test_case = 'Slice'
        testbench_path = os.path.join(script_path,
            "./testbenchs/slice.yaml")
        with open(testbench_path, "r") as stream:
            testbenchs = yaml.load(stream)
            for _t in testbenchs:
                t = testbenchs[_t]
                input_shape = [int(i) for i in t['input_shape'].split(",")]
                test_case = _test_case + _t

                x = np.random.randn(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
                y = x[0:3, 0:33, 0:15, 0:5]
                output_shape = y.shape
                starts = np.array([0, 0, 0, 0], dtype=np.int64)
                ends = np.array(y.shape, dtype=np.int64)
                axes = np.array([0, 1, 2, 3], dtype=np.int64)
                input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
                output = helper.make_tensor_value_info(
                    'output', TensorProto.FLOAT, output_shape)
                print("input", input_shape, "output", output_shape)

                #neg_node = helper.make_node(
                #    'Neg',  # node name
                #    ['input'],  # inputs
                #    ['input_neg'],  # outputs
                #)
                start_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['starts'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.INT64,
                        dims=starts.shape,
                        vals=starts.flatten().astype(int),
                    ),
                )
                ends_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['ends'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.INT64,
                        dims=ends.shape,
                        vals=ends.flatten().astype(int),
                    ),
                )
                axes_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['axes'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.INT64,
                        dims=axes.shape,
                        vals=axes.flatten().astype(int),
                    ),
                )
                node_def = helper.make_node(
                    'Slice',  # node name
                    ['input', 'starts', 'ends', 'axes'],  # inputs
                    ['output'],  # outputs
                )

                graph_def = helper.make_graph(
                    #[neg_node, start_node, ends_node, axes_node, node_def],
                    [start_node, ends_node, axes_node, node_def],
                    test_case,
                    [input],
                    [output],
                )
                model_def = helper.make_model(graph_def, producer_name=test_case)
                model_def.opset_import[0].version = 11

                input_data = np.random.rand(input_shape[0], input_shape[1],
                                input_shape[2], input_shape[3]).astype(np.float32)

                onnx.checker.check_model(model_def)
                self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Slice_3dim(self):
        test_case = 'Slice_3dim'
        input_shape = [13, 1, 256]
        output_shape = [13, 1, 128]
        x = np.random.randn(np.prod(input_shape)).reshape(
            input_shape).astype(np.float32)

        starts = np.array([0, 0, 0], dtype=np.int64)
        ends = np.array(output_shape, dtype=np.int64)
        axes = np.array([0, 1, 2], dtype=np.int64)
        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)
        start_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['starts'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=starts.shape,
                vals=starts.flatten().astype(int),
            ),
        )
        ends_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['ends'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=ends.shape,
                vals=ends.flatten().astype(int),
            ),
        )
        axes_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['axes'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=axes.shape,
                vals=axes.flatten().astype(int),
            ),
        )
        slice_def = helper.make_node(
            'Slice',  # node name
            ['input', 'starts', 'ends', 'axes'],  # inputs
            ['X1'],  # outputs
        )
        filter_data = np.random.rand(128, 128).astype(np.float32)
        filter_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['filter'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=filter_data.shape,
                vals=filter_data.flatten(),
            ),
        )
        fc_node = helper.make_node(
            'MatMul',  # node name
            ['X1', 'filter'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph(
            #[neg_node, start_node, ends_node, axes_node, node_def],
            [start_node, ends_node, axes_node, slice_def, filter_def, fc_node],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(x, model_def, test_case)

    def test_Sigmoid(self):
        test_case = 'Sigmoid'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )
        sigmoid_node = helper.make_node(
            'Sigmoid',
            ['X1'],
            ['output'],
        )
        graph_def = helper.make_graph(
            [x1_node, sigmoid_node],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)


    def test_Sub(self):
        test_case = 'Sub'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input', 'X1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [x1_def, sub_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11

        input_data = np.random.randn(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)


    def test_Sum(self):
        test_case = 'Sum'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )

        x2_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X2'],  # outputs
        )

        #test only one input
        x3_def = helper.make_node(
            'Sum',  # node name
            ['input'],  # inputs
            ['X3'],  # outputs
        )

        s1_def = helper.make_node(
            'Sum',  # node name
            ['input', 'X1'],  # inputs
            ['S1'],  # outputs
        )
        s2_def = helper.make_node(
            'Sum',  # node name
            ['X2', 'X3'],  # inputs
            ['S2'],  # outputs
        )

        #test three input
        sum_def = helper.make_node(
            'Sum',  # node name
            ['S1', 'S2'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [x1_def, x2_def, x3_def, s1_def, s2_def, sum_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Softmax(self):
        test_case = 'Softmax'
        input_shape = [4, 64, 128, 1]
        output_shape = list(input_shape)
        neg_def = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X0'],  # outputs
        )
        x1_def = helper.make_node(
            'Softmax',
            ['X0'],
            ['X1'],
            axis=1,
        )
        x2_def = helper.make_node(
            'Softmax',
            ['X0'],
            ['X2'],
            axis=2,
        )
        x3_def = helper.make_node(
            'Softmax',
            ['X0'],
            ['X3'],
            axis=3,
        )

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        X1 = helper.make_tensor_value_info(
            'X1', TensorProto.FLOAT, output_shape)
        X2 = helper.make_tensor_value_info(
            'X2', TensorProto.FLOAT, output_shape)
        X3 = helper.make_tensor_value_info(
            'X3', TensorProto.FLOAT, output_shape)
        graph_def = helper.make_graph(
            [neg_def, x1_def, x2_def, x3_def],
            test_case,
            [input],
            [X1, X2, X3],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Transpose(self):
        test_case = 'Transpose'
        transpose_order = [
            [0, 1, 2, 3],
            [0, 2, 1, 3],
            [1, 0, 2, 3],
            [1, 2, 0, 3],
            [2, 0, 1, 3],
            [2, 1, 0, 3],
        ]
        for order in transpose_order:
            input_shape = [1, 3, 27, 27]
            on = input_shape[order[0]]
            oc = input_shape[order[1]]
            oh = input_shape[order[2]]
            ow = input_shape[order[3]]
            output_shape = [on, oc, oh, ow]
            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
            output = helper.make_tensor_value_info(
                'output', TensorProto.FLOAT, output_shape)

            x1_def = helper.make_node(
                'Neg',  # node name
                ['input'],  # inputs
                ['X1'],  # outputs
            )

            #test only one input
            transpose_def = helper.make_node(
                'Transpose',  # node name
                ['X1'],  # inputs
                ['output'],  # outputs
                perm=order
            )

            graph_def = helper.make_graph(
                [x1_def, transpose_def],
                test_case,
                [input],
                [output],
            )
            model_def = helper.make_model(graph_def, producer_name=test_case)
            model_def.opset_import[0].version = 11
            onnx.checker.check_model(model_def)

            input_data = np.random.rand(input_shape[0], input_shape[1],
                            input_shape[2], input_shape[3]).astype(np.float32)

            onnx.checker.check_model(model_def)
            self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Tile(self):
        test_case = 'Tile'
        input_shape = [2,4,6,8]
        output_shape = [16,24,24,16]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        param_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['tiles'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.INT64,
                        dims=[4],
                        vals=np.array([8,6,4,2]),
                    ),
                )
        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )
        tile_node = helper.make_node(
            'Tile',
            ['X1', 'tiles'],
            ['output'],
        )
        graph_def = helper.make_graph(
            [param_node, x1_node, tile_node],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Upsample(self):
        test_case = 'Upsample'
        input_shape = [1,512,4,32]
        output_shape1 = [1,512,8,65]
        output_shape2 = [1,512,8,64]
        output_shape3 = [1,512,8,65]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output1 = helper.make_tensor_value_info(
            'output1', TensorProto.FLOAT, output_shape1)
        output2 = helper.make_tensor_value_info(
            'output2', TensorProto.FLOAT, output_shape2)
        output3 = helper.make_tensor_value_info(
            'output3', TensorProto.FLOAT, output_shape3)
        scale1_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['scale1'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[4],
                        vals=np.array([1.0,1.0,2.0,2.03125]),
                    ),
                )
        scale2_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['scale2'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[4],
                        vals=np.array([1.0,1.0,2.0,2.0]),
                    ),
                )
        scale3_node = onnx.helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=['scale3'],
                    value=onnx.helper.make_tensor(
                        name='const_tensor',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[4],
                        vals=np.array([1.0,1.0,2.0,2.03125]),
                    ),
                )
        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )
        us1_node = helper.make_node(
            'Upsample',
            ['X1', 'scale1'],
            ['output1'],
        )
        us2_node = helper.make_node(
            'Upsample',
            ['X1', 'scale2'],
            ['output2'],
        )
        us3_node = helper.make_node(
            'Upsample',
            ['X1', 'scale3'],
            ['output3'],
            mode='linear'
        )
        graph_def = helper.make_graph(
            [scale1_node, scale2_node, scale3_node, x1_node, us1_node, us2_node, us3_node],
            test_case,
            [input],
            [output1, output2, output3],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 9
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Concat(self):
        test_case = 'Concat'
        # first MUST be 4 dim for connect conv that avoid weight check
        class testbench(object):
            def __init__(self, inputs_shape, output_shape, axis, reshape_input_0=[]):
                self.inputs_shape = inputs_shape
                self.output_shape = output_shape
                self.axis = axis
                self.reshape_input_0 = reshape_input_0

            def gen_model_def(self, test_case):
                tensor_value_infos = []
                reshaped_value_infos = []

                inputs_shape = list(self.inputs_shape)

                is_not_4d = len(inputs_shape[0]) != 4
                concat_first_intput = 'X1'

                # need to reshape for conv
                if is_not_4d:
                    inputs_shape[0] = self.reshape_input_0

                for idx, shape in enumerate(inputs_shape):
                    tensor_value_infos.append(
                        helper.make_tensor_value_info('input_{}'.format(idx), TensorProto.FLOAT, shape))
                output = helper.make_tensor_value_info(
                        'output', TensorProto.FLOAT, self.output_shape)

                # conv ONLY accept 4d
                input_def_0 = helper.make_node(
                    'Neg',  # node name
                    [ 'input_0'],
                    ['X1'],  # outputs
                )

                if is_not_4d:
                    concat_first_intput = 'input_0_reshape_4_3_shaped'

                    # prepare reshape second input
                    const = np.array(self.inputs_shape[0], dtype=np.int64)
                    input_0_reshape_4_3_shape_const = helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=['input_0_reshape_4_3_shape'],
                        value=onnx.helper.make_tensor(
                            name='const_tensor',
                            data_type=onnx.TensorProto.INT64,
                            dims=const.shape,
                            vals=const.flatten().astype(int),
                        ),
                    )

                    # narrow down from 4d to 3d for concat
                    input_def_0_reshape4_3 = helper.make_node(
                        'Reshape',
                        [ 'X1', 'input_0_reshape_4_3_shape'],
                        [ concat_first_intput ],
                    )
                    reshaped_value_infos = [input_0_reshape_4_3_shape_const, input_def_0_reshape4_3]

                mlir_inputs = ['input_{}'.format(idx) for idx, v in enumerate(self.inputs_shape)]
                test_node_def = helper.make_node(
                    'Concat',  # node name
                    inputs=[concat_first_intput] + mlir_inputs[1:],  # inputs
                    axis=self.axis,
                    outputs=['output'],  # outputs
                )

                graph_def = helper.make_graph(
                    [input_def_0] + reshaped_value_infos + [test_node_def],
                    test_case,
                    tensor_value_infos,
                    [output],
                )

                model_def = helper.make_model(graph_def, producer_name=test_case)
                model_def.opset_import[0].version = 11
                return model_def

            def gen_inputs(self):
                # random data
                inputs = [np.random.rand(*_input).astype(np.float32) for _input in self.inputs_shape]
                #inputs = [np.arange(np.prod(_input)).reshape(_input).astype(np.float32) for _input in self.inputs_shape]
                # concat with 1d for mlir inference
                # https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays
                input_concated = np.concatenate([v.reshape(-1) for v in inputs])

                return inputs, input_concated

            # closure input_data
            @staticmethod
            def input_cb(input_data, reshape_input_0):
                def _input_cb(model_name, phase, input):
                    if phase == 'onnx':
                        input_data[0] = input_data[0].reshape(reshape_input_0)
                        return {'input_{}'.format(_idx): _input for _idx, _input in enumerate(input_data)}
                    elif phase == 'cvimodel':
                        cvimodel_inputs = {}
                        for _idx, _input in enumerate(input_data):
                            key = 'input_{}'.format(_idx)
                            cvimodel_inputs[key] = input[key]
                        return cvimodel_inputs
                    elif phase == 'batch':
                        return 1 # TODO: support batch concat

                return _input_cb

        testbenchs = {
            # need to assign `--batch-num`
            #'dim2_axis0': {
            #    'inputs_shape':[[1, 3072],[1, 3072]],
            #    'output_shape':[2, 3072],
            #    'axis': 0,
            #    'reshape_input_0': [1, 192, 16, 1] # for conv once dim != 4
            #},
            #'dim4_axis0': {
            #    'inputs_shape':[[1, 20, 30, 40], [6, 20, 30, 40]],
            #    'output_shape':[7, 20, 30, 40],
            #    'axis': 0,
            #},
            #'dim3_axis0': {
            #    'inputs_shape':[[1, 3072, 1],[1, 3072, 1]],
            #    'output_shape':[2, 3072, 1],
            #    'axis': 0,
            #    'reshape_input_0': [1, 192, 16, 1] # for conv once dim != 4
            #},
            'dim3_axis1_small': {
                'inputs_shape':[[1, 3072, 1],[1, 680, 1]],
                'output_shape':[1, 3752, 1],
                'axis': 1,
                'reshape_input_0': [1, 192, 16, 1] # for conv once dim != 4
            },
            # FIXME: wait backend merged
            #'dim3_axis2': {
            #    'inputs_shape':[[1, 3072, 1],[1, 3072, 11]],
            #    'output_shape':[1, 3072, 12],
            #    'axis': 2,
            #    'reshape_input_0': [1, 192, 16, 1] # for conv once dim != 4
            #},
            # FIXME: need to fix getOpGroupInputsOutputs for keep inputs order
            #'dim3_axis1': {
            #    'inputs_shape':[[1, 30720, 1],[1, 7680, 1],[1, 1920, 1],[1, 480, 1],[1, 120, 1]],
            #    'output_shape':[1, 40920, 1],
            #    'axis': 1,
            #    'reshape_input_0': [1, 1024, 30, 1] # for conv once dim != 4
            #},
            'dim4_axis1': {
                'inputs_shape':[[1, 21, 31, 4], [1, 11, 31, 4]],
                'output_shape':[1, 32, 31, 4],
                'axis': 1,
            },
            # FIXME: wait backend merged
            #'dim4_axis2': {
            #    'inputs_shape':[[1, 21, 31, 4], [1, 21, 61, 4]],
            #    'output_shape':[1, 21, 92, 4],
            #    'axis': 2,
            #},
            #'dim4_axis3': {
            #    'inputs_shape':[[1, 21, 31, 4], [1, 21, 31, 14]],
            #    'output_shape':[1, 21, 31, 18],
            #    'axis': 3,
            #},
            'dim2_axis1': {
                'inputs_shape':[[1, 400], [1, 200]],
                'output_shape':[1, 600],
                'axis': 1,
                'reshape_input_0': [1, 20, 10, 2] # for conv once dim != 4
            },
        }

        for test_case, setting in testbenchs.items():
            # init testcase
            test_case = "Concat_" + test_case
            inputs_shape = setting['inputs_shape']
            output_shape = setting['output_shape']
            axis = setting['axis']
            if 'reshape_input_0' in setting:
                reshape_input_0 = setting['reshape_input_0']
            else:
                reshape_input_0 = inputs_shape[0]

            # emit test
            tb = testbench(inputs_shape, output_shape, axis, reshape_input_0)
            model_def = tb.gen_model_def(test_case)
            input_data, input_concated = tb.gen_inputs()
            input_cb = tb.input_cb(input_data, reshape_input_0)

            onnx.checker.check_model(model_def)

            self.onnx_convert_and_infernece(input_concated, model_def, test_case, input_cb)

if __name__ == "__main__":
    os.makedirs("onnx_test", exist_ok=True)
    os.chdir("onnx_test")
    tester = ONNX_IR_TESTER()
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
