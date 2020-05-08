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


class TFModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_path):
        # TF2 use savedmodel
        self.net = tf.keras.models.load_model(model_path)

    def inference(self, input):
        return self.net.predict(input)

    def get_all_tensor(self, input_data):
        raise NotImplementError("TODO")


    def get_op_info(self):
        raise NotImplementError("TODO")
