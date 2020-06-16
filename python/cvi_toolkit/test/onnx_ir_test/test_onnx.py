#!/usr/bin/env python3

from onnx import onnx, numpy_helper
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.model.mlir_model import MLIRModel
from cvi_toolkit.utils.mlir_shell import mlir_import_calibration, mlir_tpu_quant, \
    mlir_lower_opt, mlir_build_cvimodel_no_opt, mlir_opt, \
                                        run_cvimodel
from cvi_toolkit.numpy_helper import npz_compare
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime
import numpy as np
import os
import sys
import gc

TEST_ONNX_IR = [
    "Add",
    "AveragePool",
    "GlobalMaxPool",
    "LeakyRelu",
    "LRN",
    "Max",
    "Min",
    "Neg",
    "Relu",
    "PRelu",
    "Reciprocal",
    "Slice",
    "Sub",
    "Sum",
    "Transpose",
]

NOT_SUPPORT_CMDBUF_TEST_IR = ["Relu", "Transpose"]

def make_test_calibration_table(tensors, table_name):
    # simple calibration table
    with open(table_name, 'w') as f:
        for name in tensors:
            t = 1.1 * max(np.abs(tensors[name].flatten())) + 0.01
            f.write("{} {}\n".format(name, t))

def _onnx_inference(input, model_name, input_name="input"):
    ort_session = onnxruntime.InferenceSession(model_name)
    ort_inputs = {input_name: input}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def onnx_inference(input, model_def, model_name):
    model = "{}.onnx".format(model_name)
    onnx.save(model_def, model)
    return _onnx_inference(input, model)

class ONNX_IR_TESTER(object):
    def __init__(self):
        self.converter = None
        self.cvi_model_test = True

        self.test_function = {
            "Add": self.test_Add,
            "AveragePool": self.test_AveragePool,
            "LeakyRelu": self.test_LeakyRelu,
            "LRN": self.test_LRN,
            "GlobalMaxPool": self.test_GlobalMaxPool,
            "Max": self.test_Max,
            "Min": self.test_Min,
            "Neg": self.test_Neg,
            "PRelu": self.test_PRelu,
            "Reciprocal": self.test_Reciprocal,
            "Relu": self.test_Relu,
            "Slice": self.test_Slice,
            "Sub": self.test_Sub,
            "Sum": self.test_Sum,
            "Transpose": self.test_Transpose,
        }

    def onnx_convert_and_infernece(self, input_data, model_def, model_name):
        fp32_mlir = "{}.mlir".format(model_name)
        self.converter = OnnxConverter(model_name, model_def, fp32_mlir)
        self.converter.run()
        del self.converter
        gc.collect()

        onnx_out = onnx_inference(input_data, model_def, model_name)
        self.mlir_model = None
        self.mlir_model = MLIRModel()
        self.mlir_model.load_model(fp32_mlir)
        mlir_out = self.mlir_model.inference(input_data)

        # Test output
        np.testing.assert_allclose(mlir_out, onnx_out, rtol=1e-5, atol=1e-01)

        if self.cvi_model_test:
            for i in NOT_SUPPORT_CMDBUF_TEST_IR:
                if i == model_name:
                    print("{} not support cmdbuf test!".format(model_name))
                    return

            tensors = self.mlir_model.get_all_tensor()
            # opt
            fp32_opt_mlir = "{}_opt.mlir".format(model_name)
            fp32_csv = "{}_fp32.csv".format(model_name)
            mlir_opt(fp32_mlir, fp32_opt_mlir, fp32_csv)
            table_name = "{}_cali_table".format(model_name)
            # gen cali table
            make_test_calibration_table(tensors, table_name)

            # import table
            cali_mlir = "{}_cali.mlir".format(model_name)
            int8_csv = "{}_int8.csv".format(model_name)
            ret = mlir_import_calibration(fp32_opt_mlir, cali_mlir, table_name)
            if ret < 0: raise RuntimeError("import_calibration failed")

            # quant
            quant_mlir = "{}_quant_int8.mlir".format(model_name)
            ret = mlir_tpu_quant(cali_mlir, quant_mlir, int8_csv)
            if ret < 0: raise RuntimeError("tpu_quant failed")

            # get mlir output
            del self.mlir_model
            self.mlir_model = MLIRModel()
            self.mlir_model.load_model(quant_mlir)
            mlir_int8_out = self.mlir_model.inference(input_data)
            int8_tensors = self.mlir_model.get_all_tensor()
            ref_npz = "{}_tensor_all_int8.npz".format(model_name)
            np.savez(ref_npz, **int8_tensors)

            # lower
            tg_mlir = "tg_{}.mlir".format(model_name)
            ret = mlir_lower_opt(quant_mlir, tg_mlir)
            if ret < 0: raise RuntimeError("lower_opt failed")

            # gen cvimodel
            cvimodel = "{}.cvimodel".format(model_name)
            ret = mlir_build_cvimodel_no_opt(tg_mlir, cvimodel)
            if ret < 0: raise RuntimeError("gen_cvimodel failed")

            # run cvi_model
            input_file = "{}_input.npz".format(model_name)
            output_tensor_npz = "{}_all_tensor.npz".format(model_name)
            np.savez(input_file, **{"input": tensors['input']})

            ret = run_cvimodel(input_file, cvimodel, output_tensor_npz, all_tensors=True)
            if ret < 0: raise RuntimeError("run_cvimodel failed")
            npz_compare([output_tensor_npz,ref_npz])


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
        onnx_out = _onnx_inference(input_data, model_path, input_name)

        self.mlir_model = MLIRModel()
        self.mlir_model.load_model(fp32_mlir)
        mlir_out = self.mlir_model.inference(input_data)
        # Test output
        np.testing.assert_allclose(mlir_out, onnx_out, rtol=1e-5, atol=1e-01)



        del self.mlir_model

        print("PASS")

    def test_Add(self):
        test_case = 'Add'
        input_shape = [1, 3, 27, 27]
        output_shape = [1, 3, 27, 27]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        x1_node = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['X1'],  # outputs
        )
        add_node = helper.make_node(
            'Add',  # node name
            ['input', 'X1'],  # inputs
            ['output'],  # outputs
        )
        graph_def = helper.make_graph(
            [x1_node, add_node],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
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
        input_data = np.random.rand(input_shape[0], input_shape[1],
                                    input_shape[2], input_shape[3]).astype(np.float32)
        #only support positive input for lrn
        input_data = -input_data

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
        input_data = np.random.randn(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)
        # avoid divide 0
        input_data[input_data==0] = 1
        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Relu(self):
        test_case = 'Relu'
        input_shape = [1, 3, 224, 224]
        node_def = helper.make_node(
            "Relu", # node name
            ['input'], # inputs
            ['output'], # outputs
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
        input_data = np.random.rand(input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]).astype(np.float32)

        onnx.checker.check_model(model_def)
        self.onnx_convert_and_infernece(input_data, model_def, test_case)

    def test_Slice(self):
        test_case = 'Slice'
        x = np.random.randn(1, 20, 10, 5).astype(np.float32)
        input_shape = list(x.shape)
        y = x[0:3, 0:10]
        output_shape = y.shape
        starts = np.array([0, 0], dtype=np.int64)
        ends = np.array([1, 10], dtype=np.int64)
        axes = np.array([0, 1], dtype=np.int64)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, output_shape)

        neg_node = helper.make_node(
            'Neg',  # node name
            ['input'],  # inputs
            ['input_neg'],  # outputs
        )
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
            ['input_neg', 'starts', 'ends', 'axes'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [neg_node, start_node, ends_node, axes_node, node_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)

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

        #test three input
        sum_def = helper.make_node(
            'Sum',  # node name
            ['input', 'X1', 'X2', 'X3'],  # inputs
            ['output'],  # outputs
        )

        graph_def = helper.make_graph(
            [x1_def, x2_def, x3_def, sum_def],
            test_case,
            [input],
            [output],
        )
        model_def = helper.make_model(graph_def, producer_name=test_case)
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
            onnx.checker.check_model(model_def)

            input_data = np.random.rand(input_shape[0], input_shape[1],
                            input_shape[2], input_shape[3]).astype(np.float32)

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
    elif len(sys.argv) == 1:
        pass_list = list()

        for i in TEST_ONNX_IR:
            tester.test_function.get(i)()
            pass_list.append(i)
            print("TEST {} Finish".format(i))

        print("{} PASS {}".format("="*4, "="*4))
        for i in pass_list:
            print(i)

    else:
        print("Usage: exe.py [input_shape] [model]")
        exit(-1)
