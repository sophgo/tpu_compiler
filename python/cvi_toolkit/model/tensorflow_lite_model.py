from .base_model import model_base
import os
import numpy as np
try:
    from packaging import version
    import tensorflow as tf
    # check Tf2.0
    IS_TF2 = version.parse("2.0.0") < version.parse(tf.__version__)
    if not IS_TF2:
        print("WANING, tf version is {}, we support TF2".format(
            version.parse(tf.__version__)))
except ImportError as error:
    tf = None


class TFLiteModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_file):
        self.net = tf.lite.Interpreter(model_path=model_file)
        self.net.allocate_tensors()

    def inference(self, input):
        input_details = self.net.get_input_details()
        output_details = self.net.get_output_details()
        self.net.set_tensor(input_details[0]['index'], input)

        # run
        self.net.invoke()
        return self.net.get_tensor(output_details[0]['index'])

    def get_all_tensor(self, input_data):
        raise NotImplementError("TODO")

    def get_op_info(self):
        raise NotImplementError("TODO")
