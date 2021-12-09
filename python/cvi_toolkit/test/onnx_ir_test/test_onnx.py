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
    "AveragePool1d",
    "BCastSub", # test broadcast sub
    "Concat",
    "Conv2d", # Conv with 2d case
    "ConvTranspose1d",
    # "Conv3d", # Conv with 3d case
    "DepthToSpace",
    "Einsum",
    "Einsum2",
    "Expand",
    "Expand2",
    "FullyConnected",
    "GroupFC", # test Group FC
    "Gather",
    "Gather2", # special shape
    "GlobalMaxPool",
    "GRU",
    "GRUh", # test gru output Y_h
    "LeakyRelu",
    "Log",
    "LRN",
    "LSTM",  #h c as weight
    "LSTM2", #h c as input
    "Max",
    "Min",
    "Mul",
    "MatMul",
    "Neg",
    "Pad",
    "PadReflect",
    "PRelu",
    "Relu",
    "Reduce",
    "Reduce2",
    "ReduceL2",
    "Resize",
#    "Reciprocal",
#    "Slice",
    "Slice_3dim", # test slice for 3 dims
    "Slice_step", # test slice with step
    "Sigmoid",
    "Sub",
    "Sum",
    "Softmax",
    #    "Transpose",
    "Tile",
    "Upsample",
    "Where",
]

NOT_SUPPORT_CMDBUF_TEST_IR = [""]
NOT_SUPPORT_BF16_TEST_IR = ["Relu", "LRN", "Max", "Min", "PRelu", "Reciprocal", "Transpose", "Sum", "DepthToSpace"]
NOT_SUPPORT_INT8_TEST_IR = ["Softmax"] # just for save test time

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

def onnx_inference(input, model_def, model_name, input_cb = None):
    model = "{}.onnx".format(model_name)
    onnx.save(model_def, model)
    return _onnx_inference(input, model, input_cb=input_cb)


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

class ONNX_IR_TESTER(object):
    def __init__(self):
        self.converter = None
        self.cvi_model_test = True

        self.test_function = {
            "Abs": self.test_Abs,
            "Add": self.test_Add,
            "AddConst": self.test_AddConst,
            "AveragePool": self.test_AveragePool,
            "AveragePool1d": self.test_AveragePool1d,
            "BCastSub": self.test_BCastSub,
            "Concat": self.test_Concat,
            "Conv2d": self.test_Conv2d,
            "ConvTranspose1d": self.test_ConvTranspose1d,
            "Conv3d": self.test_Conv3d,
            "DepthToSpace": self.test_DepthToSpace,
            "Einsum": self.test_Einsum,
            "Einsum2": self.test_Einsum2,
            "Expand": self.test_Expand,
            "Expand2": self.test_Expand2,
            "FullyConnected": self.test_FullyConnected,
            "GroupFC": self.test_GroupFC,
            "Gather": self.test_Gather,
            "Gather2": self.test_Gather2,
            "GlobalMaxPool": self.test_GlobalMaxPool,
            "GRU": self.test_GRU,
            "GRUh": self.test_GRUh,
            "LeakyRelu": self.test_LeakyRelu,
            "Log": self.test_Log,
            "LogSoftmax": self.test_LogSoftmax,
            "LRN": self.test_LRN,
            "LSTM": self.test_LSTM,
            "LSTM2": self.test_LSTM2,
            "Max": self.test_Max,
            "Min": self.test_Min,
            "Mul": self.test_Mul,
            "MatMul": self.test_MatMul,
            "Neg": self.test_Neg,
            "PRelu": self.test_PRelu,
            "Pad": self.test_Pad,
            "PadReflect": self.test_PadReflect,
            "Relu": self.test_Relu,
            "Reciprocal": self.test_Reciprocal,
            "Resize": self.test_Resize,
            "Slice_3dim": self.test_Slice_3dim,
            "Slice_step": self.test_Slice_step,
            "Sigmoid": self.test_Sigmoid,
            "Sub": self.test_Sub,
            "Sum": self.test_Sum,
            "Softmax": self.test_Softmax,
            "Transpose": self.test_Transpose,
            "Tile": self.test_Tile,
            "Reduce": self.test_Reduce,
            "Reduce2": self.test_Reduce2,
            "ReduceL2": self.test_ReduceL2,
            "Upsample": self.test_Upsample,
            "Where": self.test_Where,
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
        if type(input_data) != dict:
            self.converter = OnnxConverter(model_name, model_def, fp32_mlir, batch_size=0)
        else:
            self.converter = OnnxConverter(model_name, model_def, fp32_mlir)
        self.converter.run()
        del self.converter
        gc.collect()

        onnx_outs = onnx_inference(input_data, model_def, model_name, input_cb)
        num_outputs = len(onnx_outs)
        input_npz = "{}_in_fp32.npz".format(model_name)
        if isinstance(input_data, dict):
            for name in input_data:
                input_data[name] = input_data[name].astype(np.float32)
            np.savez(input_npz, **input_data)
        else:
            input_data = input_data.astype(np.float32)
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
                np.testing.assert_allclose(mlir_outs[name].flatten(), onnx_outs[onnx_name].flatten(), rtol=1e-5, atol=1e-01)
        else:
            mlir_outs = list(mlir_outs.values())[0]
            onnx_out = onnx_outs.popitem()[1]
            np.testing.assert_allclose(mlir_outs.flatten(), onnx_out.flatten(), rtol=1e-5, atol=1e-01)

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
                        print("{} not support int8 test!".format(model_name))
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

    def test_model(self, input_shape, model_path, input_name="input"):
        if isinstance(input_shape, list):
            input_shape = [int(x) for x in input_shape]
            input_shape = tuple(input_shape)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        model_name = model_path.split("/")[-1].split(".")[0]
        onnx_model = onnx.load(model_path)

        fp32_mlir = "{}.mlir".format(model_name)

        self.converter = OnnxConverter(model_name, onnx_model, fp32_mlir, batch_size=input_shape[0])
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

        input_data = np.random.rand(*input_shape).astype(np.float32)

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
        w_data = np.random.rand(*input_shape).astype(np.float32)
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
        input_data = np.random.rand(*input_shape).astype(np.float32)

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
        input_data = np.random.rand(*input_shape).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_AveragePool1d(self):
        test_case = 'AveragePool1d'
        input_data = np.random.randn(1, 3, 28).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30])

        node_def = onnx.helper.make_node(
            "AveragePool",
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3],
            strides=[1],
            pads=[2, 2],
            count_include_pad=1
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

    def test_AveragePool(self):
        test_case = 'AveragePool'
        input_data = np.random.randn(1, 3, 28, 28).astype(np.float32)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 30, 30])

        node_def = onnx.helper.make_node(
            "AveragePool",
            inputs=['input'],
            outputs=['output'],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[2, 2, 2, 2],
            count_include_pad=1
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

    def test_ConvTranspose1d(self):
        test_case = 'ConvTranspose1d'
        batch = 1
        ic =192
        oc =96
        dilations = [1]
        group = 1
        kernel_shape = [10]
        pads = [3, 3]
        strides = [5]
        output_padding = [1]
        input_shape = [batch,ic,100]
        input_data = np.random.randn(input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
        weight_shape = [ic,oc,kernel_shape[0]]
        weight_data = np.random.randn(weight_shape[0], weight_shape[1], weight_shape[2]).astype(np.float32)
        bias_data = np.random.randn(weight_shape[1]).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, list(input_data.shape))

        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [batch, oc, input_shape[-1] * strides[-1]])

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
            "ConvTranspose",
            inputs=['input', 'conv_w', 'conv_b'],
            outputs=['output'],
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dilations=dilations,
            group=group,
            output_padding=output_padding,
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
        in_shape = [1, 32, 108, 192]
        n, c, h, w = in_shape
        blocksize = 2
        # mode='CRD'
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

    def test_Einsum(self):
        # mul(1x16x28x28, 1x1x28x28) => 1x16x28x28
        test_case = 'Einsum'
        input0_shape = [1, 26, 12, 26]
        input1_shape = [12, 26, 312]
        output_shape = [1, 26, 312]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input0_shape)
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input1_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        #test only one input
        einsum_node = helper.make_node(
            'Einsum',  # node name
            ['input0', "input1"],  # inputs
            ['output'],  # outputs
            equation='bfnd,ndh->bfh',
        )

        graph_def = helper.make_graph(
            [einsum_node],
            test_case,
            [input0, input1],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 13
        input_data = {}
        input_data0 = np.random.randn(*input0_shape).astype(np.float32)
        input_data1 = np.random.randn(*input1_shape).astype(np.float32)
        input_data['input0'] = input_data0
        input_data['input1'] = input_data1

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Einsum2(self):
        # mul(1x16x28x28, 1x1x28x28) => 1x16x28x28
        test_case = 'Einsum2'
        input0_shape = [1, 26, 12, 26]
        input1_shape = [12, 26, 312]
        output_shape = [1, 26, 312]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input0_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        filter_data = np.random.rand(*input1_shape).astype(np.float32)
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
        #test only one input
        einsum_node = helper.make_node(
            'Einsum',  # node name
            ['input0', "filter"],  # inputs
            ['output'],  # outputs
            equation='bfnd,ndh->bfh',
        )

        graph_def = helper.make_graph(
            [filter_def, einsum_node],
            test_case,
            [input0],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 13
        input_data = np.random.randn(*input0_shape).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Expand(self):
        test_case = 'Expand'
        input_shape = [1, 128, 13, 1, 13, 1]
        output_shape = [1, 128, 13, 2, 13, 2]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        shape_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['new_shape'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[len(output_shape)],
                vals=np.array(output_shape, dtype=np.int64),
            ),
        )

        expand_node = helper.make_node(
            'Expand',  # node name
            ['input','new_shape'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [shape_def, expand_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Expand2(self):
        test_case = 'Expand2'
        input_shape = [1, 16]
        output_shape = [1, 16, 16]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        shape_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['new_shape'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[len(output_shape)],
                vals=np.array(output_shape, dtype=np.int64),
            ),
        )
        expand_node = helper.make_node(
            'Expand',  # node name
            ['input','new_shape'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [shape_def, expand_node],
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

    def test_Gather2(self):
        test_case = 'Gather2'
        total_tokens = 128
        input_shape = []
        output_shape = [1, 1, 256]
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
        node_def = helper.make_node(
            "LeakyRelu", # node name
            ['input'], # inputs
            ['output'], # outputs
            alpha=alpha
        )

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            test_case,
            [input],
            [output],
        )

        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(*input_shape).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Log(self):
        test_case = "Log"
        input_shape = [1, 3, 224, 224]
        node_def = helper.make_node(
            "Log",  # node name
            ['input'],  # inputs
            ['output'],  # outputs
        )

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            test_case,
            [input],
            [output],
        )

        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.clip(np.random.randn(*input_shape).astype(np.float32) * 10.0, 0.5, 8)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_LogSoftmax(self):
        test_case = "LogSoftmax"
        input_shape = [1, 100, 72]
        node_def = helper.make_node(
            "LogSoftmax",  # node name
            ['input'],  # inputs
            ['output'],  # outputs
            axis = 2,
        )

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, input_shape)
        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            test_case,
            [input],
            [output],
        )

        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(*input_shape).astype(np.float32)

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

        lrn_def = helper.make_node(
            'LRN',  # node name
            ['input'],  # inputs
            ['output'],  # outputs
            size=5,
        )

        graph_def = helper.make_graph(
            [lrn_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)

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
        Y_h = helper.make_tensor_value_info(
            'Y_h', TensorProto.FLOAT, [num_dir, batch_size, hidden_size])
        Y_c = helper.make_tensor_value_info(
            'Y_c', TensorProto.FLOAT, [num_dir, batch_size, hidden_size])
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
            outputs=['output','Y_h','Y_c'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph(
            [w_node_def, r_node_def, b_node_def, h0_node_def, c0_node_def, node_def],
            test_case,
            [input],
            [output, Y_h, Y_c],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_LSTM2(self):
        test_case = 'LSTM2'
        seq_length = 75
        batch_size = 2
        num_dir = 2
        input_size = 128
        hidden_size = 64
        direction = 'forward' if num_dir == 1 else 'bidirectional'
        input_data = np.random.rand(
            seq_length, batch_size, input_size).astype(np.float32)
        h0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        c0_data = np.random.rand(num_dir, batch_size, hidden_size).astype(np.float32)
        w_data = np.random.rand(
            num_dir, 4*hidden_size, input_size).astype(np.float32)
        r_data = np.random.rand(
            num_dir, 4*hidden_size, hidden_size).astype(np.float32)
        b_data = np.random.rand(num_dir, 8*hidden_size).astype(np.float32)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, list(input_data.shape))
        h0 = helper.make_tensor_value_info('h0', TensorProto.FLOAT, list(h0_data.shape))
        c0 = helper.make_tensor_value_info('c0', TensorProto.FLOAT, list(c0_data.shape))
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [seq_length, num_dir, batch_size, hidden_size])
        Y_h = helper.make_tensor_value_info(
            'Y_h', TensorProto.FLOAT, [num_dir, batch_size, hidden_size])
        Y_c = helper.make_tensor_value_info(
            'Y_c', TensorProto.FLOAT, [num_dir, batch_size, hidden_size])
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
        node_def = onnx.helper.make_node(
            "LSTM",
            inputs=['input', 'w', 'r', 'b', '', 'h0','c0'],
            outputs=['output','Y_h','Y_c'],
            direction=direction,
            hidden_size=hidden_size,
        )
        graph_def = helper.make_graph(
            [w_node_def, r_node_def, b_node_def, node_def],
            test_case,
            [input, h0, c0],
            [output, Y_h, Y_c],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        inputs = {
            "input":input_data,
            "h0":h0_data,
            "c0":c0_data,
        }
        self.onnx_convert_and_infernece(inputs, model_def, test_case)

    def test_Max(self):
        test_case = 'Max'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        #test three input
        max_def = helper.make_node(
            'Max',  # node name
            ['input0', 'input1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [max_def],
            test_case,
            [input0, input1],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data0 = np.random.randn(*input_shape).astype(np.float32)
        input_data1 = np.random.randn(*input_shape).astype(np.float32)
        inputs = {
            'input0':input_data0,
            'input1':input_data1,
        }
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(inputs, model_def, test_case)

    def test_Min(self):
        test_case = 'Min'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        #test four input
        min_def = helper.make_node(
            'Min',  # node name
            ['input0', 'input1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [min_def],
            test_case,
            [input0, input1],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data0 = np.random.randn(*input_shape).astype(np.float32)
        input_data1 = np.random.randn(*input_shape).astype(np.float32)
        inputs = {
            'input0':input_data0,
            'input1':input_data1,
        }
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(inputs, model_def, test_case)

    def test_Mul(self):
        # mul(1x16x28x28, 1x1x28x28) => 1x16x28x28
        test_case = 'BroadcastMul'
        input_shape = [1, 16, 28, 28]
        output_shape = [1, 16, 28, 28]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        reduce_node = helper.make_node(
            'ReduceMax',
            ['input1'],
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
            ['input2', "X2"],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [reduce_node, neg_node, mul_node],
            test_case,
            [input1, input2],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = {}
        input_data1 = np.random.randn(*input_shape).astype(np.float32)
        input_data2 = np.random.randn(*input_shape).astype(np.float32)
        input_data['input1'] = input_data1
        input_data['input2'] = input_data2

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
        input_data = np.random.rand(*input_shape).astype(np.float32)
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
        input_data = np.random.rand(*input_shape).astype(np.float32)

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
        input_data = np.random.randn(*input_shape).astype(np.float32)
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
        input_data = np.random.randn(*input_shape).astype(np.float32)
        # avoid divide 0
        input_data[input_data==0] = 1
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_PadReflect(self):
        test_case = 'PadReflect'
        input_shape = [1,  80, 100]
        output_shape = [1, 80, 106] # 6 for concat

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        pad_def = helper.make_node(
            'Pad',
            ['input'],
            ['output'],
            mode='reflect',
            pads=[0,0,3,0,0,3]
        )

        graph_def = helper.make_graph(
            [pad_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 9
        onnx.checker.check_model(model_def)

        input_data = np.random.rand(*input_shape).astype(np.float32)

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
            mode='constant'
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

        input_data = np.random.rand(*input_shape).astype(np.float32)

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

        input_data = np.random.rand(*input_shape).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Reduce(self):
        test_case = "Reduce"
        input_shape = [4, 4, 4, 16, 64]
        output_shape = [4, 4, 4, 16]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output0 = helper.make_tensor_value_info(
            'o_mean', TensorProto.FLOAT, output_shape)
        output1 = helper.make_tensor_value_info(
            'o_max', TensorProto.FLOAT, output_shape)
        output2 = helper.make_tensor_value_info(
            'o_min', TensorProto.FLOAT, output_shape)
        output3 = helper.make_tensor_value_info(
            'o_sum', TensorProto.FLOAT, output_shape)

        reduce_mean = helper.make_node(
            'ReduceMean',
            ['input'],
            ['o_mean'],
            keepdims=0,
            axes=[4],
        )
        reduce_max = helper.make_node(
            'ReduceMax',
            ['input'],
            ['o_max'],
            keepdims=0,
            axes=[4],
        )
        reduce_min = helper.make_node(
            'ReduceMin',
            ['input'],
            ['o_min'],
            keepdims=0,
            axes=[4],
        )
        reduce_sum = helper.make_node(
            'ReduceSum',
            ['input'],
            ['o_sum'],
            keepdims=0,
            axes=[4],
        )

        graph_def = helper.make_graph(
            [reduce_mean, reduce_max, reduce_min, reduce_sum],
            test_case,
            [input],
            [output0, output1, output2, output3]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(*input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Reduce2(self):
        test_case = "Reduce2"
        input_shape = [4, 4, 4, 16, 64]
        output_shape = [4, 4, 1, 1, 64]
        output_shape2 = [4, 4, 64]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output0 = helper.make_tensor_value_info(
            'o_mean', TensorProto.FLOAT, output_shape)
        output1 = helper.make_tensor_value_info(
            'o_max', TensorProto.FLOAT, output_shape)
        output2 = helper.make_tensor_value_info(
            'o_min', TensorProto.FLOAT, output_shape)
        output3 = helper.make_tensor_value_info(
            'o_sum', TensorProto.FLOAT, output_shape2)
        output4 = helper.make_tensor_value_info(
            'o_l2', TensorProto.FLOAT, output_shape)

        reduce_mean = helper.make_node(
            'ReduceMean',
            ['input'],
            ['o_mean'],
            keepdims=1,
            axes=[2,3],
        )
        reduce_max = helper.make_node(
            'ReduceMax',
            ['input'],
            ['o_max'],
            keepdims=1,
            axes=[2, 3],
        )
        reduce_min = helper.make_node(
            'ReduceMin',
            ['input'],
            ['o_min'],
            keepdims=1,
            axes=[2, 3],
        )
        reduce_sum = helper.make_node(
            'ReduceSum',
            ['input'],
            ['o_sum'],
            keepdims=0,
            axes=[2, 3],
        )

        graph_def = helper.make_graph(
            [reduce_mean, reduce_max, reduce_min, reduce_sum],
            test_case,
            [input],
            [output0, output1, output2, output3]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.randn(*input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_ReduceL2(self):
        test_case = "ReduceL2"
        input_shape = [4, 4, 4, 16, 16, 64]
        output_shape = [4, 4, 4, 64]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'o_l2', TensorProto.FLOAT, output_shape)

        reduce_l2 = helper.make_node(
            'ReduceL2',
            ['input'],
            ['o_l2'],
            keepdims=0,
            axes=[3, 4],
        )

        graph_def = helper.make_graph(
            [reduce_l2],
            test_case,
            [input],
            [output]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)


    def test_Resize(self):
        test_case = "test_Resize"
        input_shape = [1, 32, 208, 30]
        #linear
        output_shape1 = [1, 32, 416, 48] # by cpu, scale is not integer
        output_shape2 = [1, 32, 416, 90] # by npu, scale is integer
        output_shape3 = [1, 32, 104, 15] # by npu, scale is 0.5
        #nearest
        output_shape4 = [1, 32, 416, 60] # by npu, scale is integer
        output_shape5 = [1, 32, 416, 20] # by cpu

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output1 = helper.make_tensor_value_info(
            'output1', TensorProto.FLOAT, output_shape1)
        output2 = helper.make_tensor_value_info(
            'output2', TensorProto.FLOAT, output_shape2)
        output3 = helper.make_tensor_value_info(
            'output3', TensorProto.FLOAT, output_shape3)
        output4 = helper.make_tensor_value_info(
            'output4', TensorProto.FLOAT, output_shape4)
        output5 = helper.make_tensor_value_info(
            'output5', TensorProto.FLOAT, output_shape5)
        roi = np.array([], dtype=np.float32)
        scales = np.array([], dtype=np.float32)
        roi_def = onnx.helper.make_node(
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
        scales_def = onnx.helper.make_node(
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
        sizes1_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes1'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape1, dtype=np.int64),
            ),
        )
        x1_node = helper.make_node(
            'Neg',
            ['input'],
            ['X1'],
        )
        resize1_node = helper.make_node(
            'Resize',
            inputs=['X1', 'roi', 'scales', 'sizes1'],
            outputs=['output1'],
            mode='linear',
            coordinate_transformation_mode='half_pixel'
        )
        sizes2_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape2, dtype=np.int64),
            ),
        )
        resize2_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes2'],
            outputs=['output2'],
            mode='linear',
            coordinate_transformation_mode='pytorch_half_pixel'
        )
        sizes3_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes3'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape3, dtype=np.int64),
            ),
        )
        resize3_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes3'],
            outputs=['output3'],
            mode='linear',
            coordinate_transformation_mode='half_pixel'
        )
        sizes4_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes4'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape4, dtype=np.int64),
            ),
        )
        resize4_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes4'],
            outputs=['output4'],
            mode='nearest',
        )
        sizes5_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['sizes5'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=[4],
                vals=np.array(output_shape5, dtype=np.int64),
            ),
        )
        resize5_node = helper.make_node(
            'Resize',
            inputs=['input', 'roi', 'scales', 'sizes5'],
            outputs=['output5'],
            mode='nearest',
        )
        graph_def = helper.make_graph([
            x1_node, roi_def, scales_def, sizes1_def, resize1_node, sizes2_def, resize2_node, sizes3_def, resize3_node,
            sizes4_def, resize4_node, sizes5_def, resize5_node
        ], test_case, [input], [output1, output2, output3, output4, output5])

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)
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

    def test_Slice_step(self):
        test_case = 'Slice_step'
        input_shape = [1, 30, 64, 240]
        output_shape = [1, 10, 32, 60]
        x = np.random.randn(np.prod(input_shape)).reshape(
            input_shape).astype(np.float32)

        starts = np.array([0, 0, 0, 0], dtype=np.int64)
        ends = np.array(input_shape, dtype=np.int64)
        steps = np.array([1, 3, 2, 4], dtype=np.int64)
        axes = np.array([0, 1, 2, 3], dtype=np.int64)
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
        steps_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['steps'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.INT64,
                dims=steps.shape,
                vals=steps.flatten().astype(int),
            ),
        )
        slice_def = helper.make_node(
            'Slice',  # node name
            ['input', 'starts', 'ends', 'axes', 'steps'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [start_node, ends_node, axes_node, steps_node, slice_def],
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

        sigmoid_node = helper.make_node(
            'Sigmoid',
            ['input'],
            ['output'],
        )
        graph_def = helper.make_graph(
            [sigmoid_node],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)


    def test_Sub(self):
        test_case = 'Sub'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, input_shape)
        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input0', 'input1'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sub_def],
            test_case,
            [input0, input1],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        inputs = {}
        inputs['input0'] = np.random.randn(*input_shape).astype(np.float32)
        inputs['input1'] = np.random.randn(*input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(inputs, model_def, test_case)

    def test_BCastSub(self):
        test_case = 'BCastSub'
        input1_shape = [56, 27]
        input2_shape = [56, 1]
        output_shape = [56, 27]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input1_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input2_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        sub_def = helper.make_node(
            'Sub',  # node name
            ['input1', 'input2'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sub_def],
            test_case,
            [input1, input2],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11

        input1_data = np.random.randn(np.prod(input1_shape)).reshape(input1_shape).astype(np.float32)
        input2_data = np.random.randn(np.prod(input2_shape)).reshape(input2_shape).astype(np.float32)
        inputs = {
            "input1":input1_data,
            "input2":input2_data,
        }
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(inputs, model_def, test_case)

    def test_Sum(self):
        test_case = 'Sum'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, input_shape)
        input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        #test three input
        sum_def = helper.make_node(
            'Sum',  # node name
            ['input1', 'input2'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [sum_def],
            test_case,
            [input1, input2],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)

        input_data = {}
        input_data1 = np.random.rand(*input_shape).astype(np.float32)
        input_data2 = np.random.rand(*input_shape).astype(np.float32)
        input_data['input1'] = input_data1
        input_data['input2'] = input_data2

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Softmax(self):
        test_case = 'Softmax'
        input_shape = [4, 64, 128, 1]
        output_shape = list(input_shape)
        x1_def = helper.make_node(
            'Softmax',
            ['input'],
            ['X1'],
            axis=1,
        )
        x2_def = helper.make_node(
            'Softmax',
            ['input'],
            ['X2'],
            axis=2,
        )
        x3_def = helper.make_node(
            'Softmax',
            ['input'],
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
            [x1_def, x2_def, x3_def],
            test_case,
            [input],
            [X1, X2, X3],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        onnx.checker.check_model(model_def)
        input_data = np.random.rand(*input_shape).astype(np.float32)
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

            #test only one input
            transpose_def = helper.make_node(
                'Transpose',  # node name
                ['input'],  # inputs
                ['output'],  # outputs
                perm=order
            )

            graph_def = helper.make_graph(
                [transpose_def],
                test_case,
                [input],
                [output],
            )
            model_def = helper.make_model(graph_def, producer_name=test_case)
            model_def.opset_import[0].version = 11
            onnx.checker.check_model(model_def)

            input_data = np.random.rand(*input_shape).astype(np.float32)

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
        tile_node = helper.make_node(
            'Tile',
            ['input', 'tiles'],
            ['output'],
        )
        graph_def = helper.make_graph(
            [param_node, tile_node],
            test_case,
            [input],
            [output],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)

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
                        vals=np.array([o/i for o,i in zip(output_shape1,input_shape)]),
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
                        vals=np.array([o/i for o,i in zip(output_shape2,input_shape)]),
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
                        vals=np.array([o/i for o,i in zip(output_shape3,input_shape)]),
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
            [x1_node, scale1_node, scale2_node, scale3_node, us1_node, us2_node, us3_node],
            test_case,
            [input],
            [output1, output2, output3],
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 9
        input_data = np.random.rand(*input_shape).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Where(self):
        test_case = "Where"
        const_shape = [1]
        const_data = np.array([0.5]).astype(np.float32)
        testbench = [
            {
              'input_shape': [1, 3, 4, 4],
              'condition_shape': [1, 3, 1, 1],
              'condition_data': np.array([[[[1]]],[[[1]]],[[[0]]]])
            },
            {
              'input_shape': [1, 3, 4],
              'condition_shape': [1, 3, 1],
              'condition_data': np.array([[[[1]]],[[[1]]],[[[0]]]])
            },
        ]

        idx = 1

        input_shape = testbench[idx]['input_shape']
        condition_shape = testbench[idx]['condition_shape']
        condition_data = testbench[idx]['condition_data']

        real_test = True
        #real_test = False
        if real_test:
            # real case
            input_shape = [1, 40, 224]
            output_shape = input_shape
            condition_shape = [1, 40, 224]
            condition_data = np.zeros(condition_shape)
            # select top half
            #array([[[1., 1., 0., 0.],
            #        [1., 1., 0., 0.],
            #        [1., 1., 0., 0.],
            #        [1., 1., 0., 0.]],
            condition_data[:,:,:100] = 1

        output_shape = input_shape
        condition_data = condition_data.astype(np.bool)
        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        condition_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['condition'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.BOOL,
                dims=condition_shape,
                vals=condition_data.flatten(),
            ),
        )

        masked_fill_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const'],
            value=onnx.helper.make_tensor(
                name='const_tensor_1',
                data_type=onnx.TensorProto.FLOAT,
                dims=const_data.shape,
                vals=const_data,
            ),
        )
        where_node = helper.make_node(
            'Where',
            ['condition', 'const', 'input'],
            ['output'],
        )

        graph_def = helper.make_graph(
            [condition_node_def, masked_fill_node_def, where_node],
            test_case,
            [input],
            [output]
        )

        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Concat(self):
        test_case = 'Concat'
        input_shape = [1, 192, 256]
        x_shape = [1, 192, 16]
        output_shape = [1, 192, 288]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
        x1_data = np.random.randn(*x_shape).astype(np.float32)
        x1_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['X1'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=x_shape,
                vals=x1_data.flatten(),
            ),
        )
        x2_data = np.random.randn(*x_shape).astype(np.float32)
        x2_node_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['X2'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=x_shape,
                vals=x2_data.flatten(),
            ),
        )
        concat_node = helper.make_node(
            'Concat',
            ['input', 'X1', 'X2'],
            ['output'],
            axis = -1,
        )
        graph_def = helper.make_graph(
            [x1_node_def, x2_node_def, concat_node],
            test_case,
            [input],
            [output]
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
        model_def.opset_import[0].version = 11
        input_data = np.random.rand(*input_shape).astype(np.float32)
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)


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
        tester.set_quant_mode(mode="bf16")
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
        print("Onnx test result:")
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
