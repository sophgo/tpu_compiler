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
        output_names = [l.name for l in self.net.layers]
        all_tensor_model = tf.keras.Model(inputs=self.net.inputs, outputs=self.net.outputs + [l.output for l in self.net.layers])
        output_values = all_tensor_model.predict(input_data)
        all_tensor_model_names = [l.name for l in all_tensor_model.layers]
        # all_tensor_model_outputs = [l.output for l in all_tensor_model.layers]
        all_tensor_dict = dict(
            zip(all_tensor_model_names, output_values[len(self.net.outputs):]))
        # data [output1, output2 ... input1 ... ]
        all_tensor_dict['input'] = output_values[len(self.net.outputs)]
        return all_tensor_dict

    def get_all_weights(self):
        weight_tensor = dict()
        for layer in self.net.layers:
            config = layer.get_config()
            ws = layer.get_weights()
            if len(ws) != 0:
                for idx, w in enumerate(ws):
                    weight_tensor['{}_{}'.format(layer.name, idx)] = w
        return weight_tensor

    def get_op_info(self):
        return [l.name for l in self.net.layers]
