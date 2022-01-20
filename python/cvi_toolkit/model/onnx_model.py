from .base_model import model_base
import onnxruntime
import onnx
from onnx import helper
import os
import numpy as np

class OnnxModel(model_base):
    def __init__(self):
        self.net = None

    def get_shape(self, sess):
        """
            Onnxruntime output shape maybe wrong, we print here to check
        """
        for output in sess.get_outputs():
            print("output name {}\n".format(output.name))
            print("output shape {}".format(output.shape))


    def load_model(self, model_file, wegiht_file=None):
        self.model_file = model_file
        self.net = onnxruntime.InferenceSession(model_file)
        self.onnx_model = onnx.load(model_file)

    def inference(self, inputs):
        return self._infernece(inputs)

    def _infernece(self, inputs, onnx_model=None):
        if onnx_model:
            ort_session = onnxruntime.InferenceSession(onnx_model)
            # self.get_shape(ort_session)
            ort_outs = ort_session.run(None, inputs)
        else:
            ort_outs = self.net.run(None, inputs)
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

    def get_inputs(self, ):
        return self.net.get_inputs()

    def get_op_info(self):
        raise RuntimeError("Todo")