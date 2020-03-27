from .base_model import model_base
import caffe
import numpy as np

class CaffeModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_file, wegiht_file):
        self.net = caffe.Net(model_file, wegiht_file, caffe.TEST)

    def inference(self, input):
        out = self.net.forward_all(**{self.net.inputs[0]: input})
        return out[self.net.outputs[0]]

    def get_all_tensor(self, input_data, npz_file):

        print("Save Blobs: ", npz_file)
        blobs_dict = {}
        for name, layer in self.net.layer_dict.items():
            print("layer : " + str(name))
            print("  type = " + str(layer.type))
            print("  top -> " + str(self.net.top_names[name]))
            if layer.type == "Split":
                print("  skip Split")
                continue
            if layer.type == "Slice":
                print(" skip Slice")
                continue
            assert(len(self.net.top_names[name]) == 1)
            if layer.type == "Input":
                blobs_dict[name] = input_data
                continue
            #out = self.forward(None, prev_name, name, **{prev_name: prev_data})
            out = self.net.forward(None, None, name, **{self.net.inputs[0]: input_data})
            blobs_dict[name] = out[self.net.top_names[name][0]].copy()
        np.savez(npz_file, **blobs_dict)
