try:
    from packaging import version
    import tensorflow as tf
    # check Tf2.0
    IS_TF2 = version.parse("2.0.0") < version.parse(tf.__version__)

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_session = tf.compat.v1.Session
    extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
except ImportError as error:
    tf = None

from .log_setting import setup_logger
import logging
logger = setup_logger('root')
log_flag = logger.level <= logging.INFO


def get_tf_node_attr(node, name):
    """Parser TF node attribute."""
    return node.get_attr(name)


def get_tf_shape_attr(node):
    """Get shape from tensorflow attr "shape"."""
    dims = None
    try:
        shape = get_tf_node_attr(node, "shape")
        if not shape.unknown_rank:
            dims = [int(d.size) for d in shape.dim]
    except:
        pass
    return dims

def tf_reload_graph(tf_graph):
    """Invoke tensorflow cpp shape inference by reloading graph_def."""
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    with tf.Graph().as_default() as inferred_graph:
        tf.import_graph_def(graph_def, name="")
    return inferred_graph

def infer_shape_for_graph(tf_graph):
    """
    Infer shape for Tensorflow ops.
    Tensorflow explicitly sets shape for some ops in python code, such as Switch, Merge and TensorArrayGather.
    These shapes may be lost after freezing TF graph to graph_def without add_shapes=True.
    To bring these shapes back, we implement our own shape inference for these control flow ops based on one assumption:
    **outputs of Merge op have the same shape (at least the same rank) of its inputs**.
    With this assumption, our shape inference can handle:
        1. in tf.cond, outputs of two branches have the same rank.
        2. in tf.while_loop, loop variables don't change their rank.
    """
    shape_updated = True
    while shape_updated:
        shape_updated = False
        for o in tf_graph.get_operations():
            updated = infer_shape_for_op(o)
            if updated:
                shape_updated = True
        if shape_updated:
            tf_graph = tf_reload_graph(tf_graph)
    return tf_graph

def infer_shape(tf_graph, shape_override=None):
    """Infer shape for TF graph with shape_override set first."""
    if shape_override:
        logger.info("Apply shape override:")
        for name, shape in shape_override.items():
            logger.info("\tSet {} shape to {}".format(name, shape))
            tf_graph.get_tensor_by_name(name).set_shape(shape)
        tf_graph = tf_reload_graph(tf_graph)

    tf_graph = infer_shape_for_graph(tf_graph)

    op_outputs_with_none_shape = check_shape_for_tf_graph(tf_graph)
    if op_outputs_with_none_shape:
        for op, outs in op_outputs_with_none_shape.items():
            logger.warning(
                "Cannot infer shape for {}: {}".format(
                op, ",".join(outs))
            )
        tf_graph = infer_shape_for_graph_legacy(tf_graph)

    return tf_graph

def tf_node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def tf_optimize_grappler(input_names, output_names, graph_def, fold_constant=None):
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2
    from tensorflow.python.grappler import tf_optimizer as tf_opt

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    # TODO: if we turn on pruning, grappler removes some identities that the tf-1.x lstm rewriter
    #   depends on so for now don't turn this on.
    rewrite_options.optimizers[:] = [
        # 'pruning', 'constfold', 'arithmetic', 'dependency', 'function',
        'constfold', 'function'
    ]
    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in input_names + output_names:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
    graph_def = tf_opt.OptimizeGraph(config, meta_graph)
    return graph_def


def tf_optimize(input_names, output_names, graph_def, fold_constant=True):
    """Extract inference subgraph and optimize graph."""
    assert(isinstance(input_names, list))
    assert(isinstance(output_names, list))

    needed_names = [tf_node_name(i) for i in input_names] + \
                   [tf_node_name(i) for i in output_names]
    graph_def = extract_sub_graph(graph_def, needed_names)
    if fold_constant:
        graph_def = tf_optimize_grappler(
            input_names, output_names, graph_def, fold_constant)
    return graph_def


def inputs_without_resource(sess, input_names):
    try:
        new_input_names = []
        for n in input_names:
            t = sess.graph.get_tensor_by_name(n)
            if t.dtype != tf.dtypes.resource:
                new_input_names.append(n)
        input_names = new_input_names
    except:
        pass
    return input_names


def from_function(func, input_names, output_names):
    frozen_func = convert_variables_to_constants_v2(
        func, lower_control_flow=False)
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

    tf_reset_default_graph()
    with tf_session() as sess:
        tf.import_graph_def(graph_def, name='')
        input_names = inputs_without_resource(sess, input_names)
        graph_def = tf_optimize(input_names, output_names, graph_def)
    return graph_def


def from_saved_model(model_path):
    """Load tensorflow graph from saved_model."""
    imported = tf.saved_model.load(model_path, tags="serve")
    signature_def = "serving_default"
    concrete_func = imported.signatures[signature_def]
    inputs = [
        tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
    outputs = [
        tensor.name for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]
    tf_reset_default_graph()
    frozen_graph = from_function(
        concrete_func, inputs, outputs)

    tf_reset_default_graph()

    return frozen_graph, inputs, outputs

