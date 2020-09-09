from .base_model import model_base
import tensorflow as tf
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
    tf_session = tf.compat.v1.Session
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
except ImportError as error:
    tf = None

from ..utils.tf_utils import from_saved_model, tf_node_name


class TFModel(model_base):
    def __init__(self):
        self.net = None

    def load_model(self, model_path):
        # TF2 use savedmodel
        self.net = tf.saved_model.load(model_path)
        self.tf_graph, self.inputs, self.outputs = from_saved_model(model_path)


    def inference(self, input):
        return self.net(input)

    def get_all_tensor(self, input_data):
        all_op_info = self.get_op_info()

        valid_op = all_op_info[:]
        tf_reset_default_graph()
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.tf_graph, name='')
        with tf_session(graph=graph) as sess:
            output_tensor_list = list()
            input_tensor = sess.graph.get_tensor_by_name(self.inputs[0])
            for op in all_op_info:
                try:
                    output_tensor_list.append(
                        sess.graph.get_tensor_by_name("{}:0".format(op)))
                except KeyError as key_err:
                    print("skip op {}".format(op))
                    valid_op.remove(op)
            output = sess.run(tuple(output_tensor_list), feed_dict={input_tensor: input_data})

        all_tensor_dict = dict(zip(valid_op, output))
        all_tensor_dict['input'] = all_tensor_dict[tf_node_name(self.inputs[0])]
        all_tensor_dict['output'] = output[-1]
        tf_reset_default_graph()

        return all_tensor_dict

    def get_all_weights(self):
        pass

    def get_op_info(self):
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(self.tf_graph, name='')
        with tf_session(graph=graph):
            node_list = graph.get_operations()
        return [i.name for i in node_list]
