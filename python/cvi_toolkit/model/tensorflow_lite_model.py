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
        input_details = self.net.get_input_details()
        self.net.set_tensor(input_details[0]['index'], input_data)
        self.net.invoke()
        all_tensor_dict = dict()
        for t in self.net.get_tensor_details():

            tensor_data = self.net.get_tensor(t['index'])
            tensor_data = tensor_data.reshape(t['shape'])
            print(t)
            if len(t['shape']) == 4:
                # Transpose NHWC tensor to NCHW
                tensor_data = np.transpose(tensor_data, (0, 3, 1, 2))
            all_tensor_dict[str(t['index'])] = tensor_data
        print(input_details[0]['index'])
        all_tensor_dict['input'] = all_tensor_dict[str(input_details[0]['index'])]
        return all_tensor_dict


    def get_op_info(self):
        raise NotImplementError("TODO")
