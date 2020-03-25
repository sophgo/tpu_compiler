from .base_model import model_base
import caffe

class CaffeModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_file, wegiht_file):
        self.net = caffe.Net(model_file, wegiht_file, caffe.TEST)

    def inference(self, input):
        out = self.net.forward_all(**{self.net.inputs[0]: input})
        return out[self.net.outputs[0]]