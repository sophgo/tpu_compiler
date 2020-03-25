from .base_model import model_base
import onnxruntime


class OnnxModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_file, wegiht_file=None):
        self.net = onnxruntime.InferenceSession(model_file)

    def inference(self, input):
        ort_inputs = {self.net.get_inputs()[0].name: input}
        ort_outs = self.net.run(None, ort_inputs)
        return ort_outs[0]

