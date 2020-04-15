from .base_model import model_base
import onnxruntime
import onnx
from onnx import helper
import os
import numpy as np

class OnnxModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_file, wegiht_file=None):
        self.model_file = model_file
        self.net = onnxruntime.InferenceSession(model_file)
        self.onnx_model = onnx.load(model_file)

    def inference(self, input):
        return self._infernece(input)[0]

    def _infernece(self, input, onnx_model=None):
        if onnx_model:
            ort_session = onnxruntime.InferenceSession(onnx_model)
            ort_inputs = {ort_session.get_inputs()[0].name: input}
            ort_outs = ort_session.run(None, ort_inputs)
        else:
            ort_inputs = {self.net.get_inputs()[0].name: input}
            ort_outs = self.net.run(None, ort_inputs)
        return ort_outs

    def get_all_tensor(self, input_data):
        output_keys = ['output']
        onnx_model = self.onnx_model
        for node in onnx_model.graph.node:
            _intermediate_tensor_name = list(node.output)
            intermediate_tensor_name = ",".join(_intermediate_tensor_name)
            intermediate_layer_value_info = helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            onnx_model.graph.output.append(intermediate_layer_value_info)
            output_keys.append(intermediate_layer_value_info.name + '_' + node.op_type)

        dump_all_onnx = "all_{}".format(self.model_file.split("/")[-1])
        onnx.save(onnx_model, dump_all_onnx)

        print("dump multi-output onnx all tensor at ", dump_all_onnx)
         # dump all inferneced tensor
        ort_outs = self._infernece(input_data, onnx_model=dump_all_onnx)
        output_dict = dict(zip(output_keys, map(np.ndarray.flatten, ort_outs)))

        return output_dict

    def get_op_info(self):
        raise RuntimeError("Todo")