#!/usr/bin/env python3
import os
import sys
import abc
import onnx, onnxruntime
import numpy as np
import argparse
from cvi_toolkit.utils.version import declare_toolchain_version
from cvi_toolkit.utils.log_setting import setup_logger
from cvi_toolkit.model import CaffeModel
from cvi_toolkit.model.ModelFactory import ModelFactory
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.transform.tflite_converter_int8 import TFLiteConverter as TFLiteInt8Converter
from cvi_toolkit.transform.tensorflow_converter import TFConverter
from cvi_toolkit.transform.caffe_converter import CaffeConverter
from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.utils.mlir_shell import *
from cvi_toolkit.utils.intermediate_file import IntermediateFile
from cvi_toolkit.utils.mlir_parser import MlirParser

logger = setup_logger('root', log_level="INFO")

class ModelTransformTool(object):
    def __init__(self, model_name, batch_size, preprocessor):
        self.model_name = model_name
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.preprocess_args = preprocessor.to_dict()
        self.op_info_csv = IntermediateFile(self.model_name, 'op_info.csv', True)

    def cleanup(self):
        IntermediateFile.cleanup()

    def model_transform(self, mlir_file):
        self.mlir_file = IntermediateFile('', mlir_file)
        tmp_mlir_file = IntermediateFile(self.model_name, 'fp32.mlir.tmp', False)
        self._transform_(str(tmp_mlir_file))
        ret = mlir_opt(str(tmp_mlir_file), str(self.mlir_file), str(self.op_info_csv))
        if ret != 0:
            raise RuntimeError("mlir graph optimize fail")
        self.ppa = preprocess()
        self.input_num = self.ppa.get_input_num(str(self.mlir_file))

    @staticmethod
    def _is_npz(image):
        return True if image.split('.')[-1] == 'npz' else False

    @staticmethod
    def _is_npy(image):
        return True if image.split('.')[-1] == 'npy' else False

    def model_validate(self, image, tolerance, excepts):
        in_fp32_npz = IntermediateFile(self.model_name, 'in_fp32.npz', False)
        inputs = {}
        images = image.split(',')
        if len(images) == 1 and self._is_npz(image):
            npz_in = np.load(image)
            np.savez(str(in_fp32_npz), **npz_in)
            for name in npz_in.files:
                inputs[name] = npz_in[name]
        else:
            for i in range(self.input_num):
                file = os.path.expanduser(images[i])
                self.ppa.load_config(str(self.mlir_file), i)
                # prepare input tensor
                if self._is_npy(file):
                    inputs[self.ppa.input_name] = np.load(file)
                else:
                    inputs[self.ppa.input_name] = self.preprocessor.run(file, batch=self.batch_size)
        np.savez(str(in_fp32_npz), **inputs)

        # original model inference to get blobs of all layer
        all_blobs = self._inference_(inputs)
        blobs_npz = IntermediateFile(self.model_name, 'blobs.npz', False)
        np.savez(str(blobs_npz), **all_blobs)

        # inference of mlir model
        blobs_interp_npz = IntermediateFile(self.model_name, 'blobs_interp.npz', False)
        ret = mlir_inference(str(self.mlir_file), str(in_fp32_npz), None,
                             str(blobs_interp_npz))
        if ret != 0:
            raise RuntimeError("interpret fail")

        # compare all blobs layer by layers
        ret = fp32_blobs_compare(str(blobs_interp_npz), str(blobs_npz),
                                 str(self.op_info_csv), tolerance,
                                 excepts=excepts)
        if ret != 0:
            raise RuntimeError("validate fail")

    @abc.abstractmethod
    def _inference_(self):
        pass

    @abc.abstractmethod
    def _transoform_(self):
        pass


class CaffeModelTransformTool(ModelTransformTool):
    def __init__(self, model_name, prototxt, caffemodel,
                 batch_size, preprocessor):
        super().__init__(model_name, batch_size, preprocessor)
        self.prototxt = prototxt
        self.caffemodel = caffemodel

    def _inference_(self, inputs):
        caffemodel = CaffeModel()
        caffemodel.load_model(self.prototxt, self.caffemodel)
        caffemodel.inference(inputs) # just for reshape ?
        return caffemodel.get_all_tensor(inputs, False)

    def _transform_(self, mlir_file):
        cvt = CaffeConverter(self.model_name, self.prototxt, self.caffemodel,
                             mlir_file, batch_size=self.batch_size,
                             preprocess_args=self.preprocess_args)
        cvt.run()


class OnnxModelTransformTool(ModelTransformTool):
    def __init__(self, model_name, onnx_model, batch_size, preprocessor):
        super().__init__(model_name, batch_size, preprocessor)
        self.onnx_model = onnx_model

    def generate_tmp_onnx(self):
        # for dump all activations
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        model = onnx.load(self.onnx_model)
        no_list = [
            "Cast", "Shape", "Unsqueeze", "Gather", "Split", "Constant", "GRU",
            # remove layernorm's sub ops, or it will crash in onnxruntime.
            "Div", "Sqrt", "Add", "ReduceMean", "Pow", "Sub", "Mul", "LSTM"
        ]

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            if x.op_type in no_list:
                continue
            _intermediate_tensor_name = list(x.output)
            intermediate_tensor_name = ",".join(_intermediate_tensor_name)
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            model.graph.output.append(intermediate_layer_value_info)
            output_keys.append(
                intermediate_layer_value_info.name + '_' + x.op_type)
        dump_all_tensors_onnx = IntermediateFile(self.model_name, 'dump_all_tensors.onnx', False)
        onnx.save(model, str(dump_all_tensors_onnx))
        return output_keys, dump_all_tensors_onnx

    def _fill_inputs(self, ort_session, inputs):
        inodes = ort_session.get_inputs()
        data = {}
        for i in range(len(inodes)):
            name = inodes[i].name
            dtype = np.int64 if inodes[i].type == 'tensor(int64)' \
                             else np.float32
            data[name] = inputs[name].astype(dtype)
        return data

    def _inference_(self, inputs):
        output_keys, tmp_onnx = self.generate_tmp_onnx()
        ort_session = onnxruntime.InferenceSession(str(tmp_onnx))
        ort_inputs = self._fill_inputs(ort_session, inputs)
        ort_outs = ort_session.run(None, ort_inputs)
        output_num = len(ort_outs) - len(output_keys)
        ort_outs = ort_outs[output_num:]
        return dict(zip(output_keys, map(np.ndarray.flatten, ort_outs)))

    def _transform_(self, mlir_file):
        model = onnx.load(self.onnx_model)
        cvt = OnnxConverter(self.model_name, model, mlir_file,
                            batch_size=self.batch_size,
                            preprocess_args=self.preprocess_args)
        cvt.run()


class TFModelTransformTool(ModelTransformTool):
    def __init__(self, model_name, model_def, batch_size, preprocessor):
        super().__init__(model_name, batch_size, preprocessor)
        self.model_def = model_def

    def _inference_(self, inputs):
        net = ModelFactory()
        net.load_model("tensorflow", model_file=self.model_def)
        _,input = inputs.popitem()
        transposed_inputs = np.transpose(input, [0, 2, 3, 1])
        net.inference(transposed_inputs)
        all_blobs = net.get_all_tensor(transposed_inputs)
        npz_out = {}
        for k, v in all_blobs.items():
            if len(v.shape) != 4:
                npz_out[k] = v
            else:
                npz_out[k] = np.ascontiguousarray(np.transpose(v, (0, 3, 1, 2)))
        return npz_out

    def _transform_(self, mlir_file):
        cvt = TFConverter(self.model_name, self.model_def, mlir_file,
                          batch_size=self.batch_size,
                          preprocess_args=self.preprocess_args)
        cvt.run()


class TFLiteInt8ModelTransformTool(ModelTransformTool):
    def __init__(self, model_name, model_def, batch_size, preprocessor):
        super().__init__(model_name, batch_size, preprocessor)
        self.model_def = model_def

    def _inference_(self, inputs):
        net = ModelFactory()
        net.load_model("tflite_int8", model_file=self.model_def)
        _,input = inputs.popitem()
        transposed_inputs = np.transpose(input, [0, 2, 3, 1])
        net.inference(transposed_inputs)
        all_blobs = net.get_all_tensor(transposed_inputs)
        npz_out = {}
        for k, v in all_blobs.items():
            if len(v.shape) != 4:
                npz_out[k] = v
            else:
                npz_out[k] = np.ascontiguousarray(np.transpose(v, (0, 3, 1, 2)))
        return npz_out

    def _transform_(self, mlir_file):
        assert(self.batch_size == 1)
        cvt = TFLiteInt8Converter(self.model_name, self.model_def, mlir_file,
                                  preprocess_args=self.preprocess_args)
        cvt.run()

def get_model_transform(args):
    preprocessor = preprocess()
    preprocessor.config(**vars(args))

    if args.model_type == 'caffe':
        tool = CaffeModelTransformTool(args.model_name, args.model_def, args.model_data,
                                       args.batch_size, preprocessor)
    elif args.model_type == 'onnx':
        tool = OnnxModelTransformTool(args.model_name, args.model_def,
                                      args.batch_size, preprocessor)
    elif args.model_type == 'tensorflow':
        tool = TFModelTransformTool(args.model_name, args.model_def,
                                    args.batch_size, preprocessor)
    else: # tflite_int8
        tool = TFLiteInt8ModelTransformTool(args.model_name, args.model_def,
                                            args.batch_size, preprocessor)
    return tool


class Mlir2Graph:
    def __init__(self, mlir):
        self.parser = MlirParser(mlir)
        self.to_graph(mlir + ".pb")

    def create_input_tvis(self):
        ops = self.parser.get_all_ops()
        tvis = []
        for op in ops:
            if op.type == 'tpu.input':
                tvi = onnx.helper.make_tensor_value_info(
                            op.name, onnx.TensorProto.FLOAT, op.shape)
                tvis.append(tvi)
        return tvis

    def create_output_tvis(self):
        ops = self.parser.get_all_ops()
        outputs = set(self.parser.get_output_op_names())
        tvis = []
        for op in ops:
            if op.name in outputs:
                tvi = onnx.helper.make_tensor_value_info(
                            op.name, onnx.TensorProto.FLOAT, op.shape)
                tvis.append(tvi)
        return tvis

    def to_graph(self, graph):
        inputs = self.create_input_tvis()
        outputs = self.create_output_tvis()
        nodes = []
        for op in self.parser.get_all_ops():
            if op.type == "tpu.input":
                continue
            node = onnx.helper.make_node(
                op.type, op.opds, [op.name], shape=op.shape, **op.attrs)
            nodes.append(node)

        graph_def = onnx.helper.make_graph(nodes, 'mlir', inputs, outputs)
        model_def = onnx.helper.make_model(graph_def, producer_name="cvitek")
        onnx.save(model_def, graph)


if __name__ == '__main__':
    declare_toolchain_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input image for inference")
    parser.add_argument("--model_def", help="model definition file.")
    parser.add_argument("--model_data", help="caffemodel, only for caffe model")
    parser.add_argument("--model_name", help="model name")
    parser.add_argument("--model_type", choices=['caffe', 'onnx', 'tensorflow'], help="model_type")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--tolerance", default='0.99,0.99,0.98',
                        help="minimum similarity tolerance to model transform")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--mlir", required=True, help="output mlir model file")
    parser.add_argument("--graph", action='store_true', help="generate graph to pb file")
    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    tool = get_model_transform(args)
    tool.model_transform(args.mlir)
    tool.model_validate(args.image, args.tolerance, args.excepts)
    if args.graph:
        Mlir2Graph(args.mlir)
    tool.cleanup()
