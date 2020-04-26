from .base_model import model_base
import os
os.environ['GLOG_minloglevel'] = '3'
import caffe
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CaffeModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_file, wegiht_file):
        self.net = caffe.Net(model_file, wegiht_file, caffe.TEST)

    def inference(self, input):
        # reshape to multi-batch blobs
        in_ = self.net.inputs[0]
        self.net.blobs[in_].reshape(input.data.shape[0], self.net.blobs[in_].data.shape[1],  self.net.blobs[in_].data.shape[2],  self.net.blobs[in_].data.shape[3])

        out = self.net.forward_all(**{self.net.inputs[0]: input})
        return out[self.net.outputs[0]]

    def get_input_shape(self):
        if not self.net:
            print("Not init caffe model")
            return None
        else:
            in_ = self.net.inputs[0]
            return "{},{}".format(self.net.blobs[in_].data.shape[2], self.net.blobs[in_].data.shape[3])
    def get_all_tensor(self, input_data=None):
        if input_data is None:
            print("[Warning] Caffe model get all tensor need input data")
            return None

        blobs_dict = {}
        blobs_dict['raw_data'] = input_data
        for name, layer in self.net.layer_dict.items():
            msg = "layer : {}\n\ttype = {} \n\ttop -> {}".format(name, layer.type, self.net.top_names[name])
            logger.debug(msg)
            if layer.type == "Split":
                continue
            if layer.type == "Slice":
                continue
            assert(len(self.net.top_names[name]) == 1)
            if layer.type == "Input":
                blobs_dict[name] = input_data
                continue
            out = self.net.forward(None, None, name, **{self.net.inputs[0]: input_data})
            blobs_dict[name] = out[self.net.top_names[name][0]].copy()
        return blobs_dict

    def get_op_info(self):
        return self.net.layer_dict.items()
