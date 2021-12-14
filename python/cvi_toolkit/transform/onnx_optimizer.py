from typing import cast
import onnx
import copy
import torch
import numpy as np
import onnxruntime as rt
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
from collections import OrderedDict
from numbers import Number
from onnx import TensorProto, mapping


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        return str_list
    elif attr_proto.name:
        name_list = list(attr_proto.name)
        return name_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}

def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)

def get_attr(attrs, name):
    attrs = dict([(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr)))
                  for attr in attrs])
    return attrs[name]

def fixed_point(fun):
    flag = fun()
    while True:
        if flag:
            flag = fun()
            continue
        break

def dump_model(model, name="opt.onnx"):
    data = model.SerializeToString()
    with open(name, "wb") as file:
        file.write(data)


class PesudoNode(object):
    def __init__(self, op_type, input=None, output=None, attr_key_map=None,
                 const_value=None, default=None, constraint=None):
        if input is None:
            input = []
        if output is None:
            output= []
        if attr_key_map is None:
            attr_key_map = []
        if default is None:
            default = {}
        self.op_type = op_type
        self.input = input
        self.output = output
        # get /set attr and map to new key
        self.attr_key_map = attr_key_map
        # for constant node
        self.const_value = const_value
        self.default = default
        # for broat cast or other constraint
        self.constraint = constraint


class RedundanciesOps(object):
    def __init__(self, pattern_input, pattern_output,  attr, redundancies_ops):
        self.pattern_input = pattern_input
        self.pattern_output = pattern_output
        self.redundancies_ops = redundancies_ops
        self.attr = attr


class FoldUnfoldInfo(object):
    def __init__(self, src_node, trg_node):
        self.src_node = src_node
        self.trg_node = trg_node


class Form_Deform(object):
    # support form/deform multi input single output op
    def __init__(self, model):
        self.op_list = []
        self.nodes = model.graph.node
        self.weight = model.graph.initializer
        # store node shape
        self.shape_info = [info for info in model.graph.value_info]
        self.shape_info.extend(model.graph.output)
        self.shape_info = {info.name: [i.dim_value for i in info.type.tensor_type.shape.dim if i.dim_value > 0]
                            for info in self.shape_info}
        self.tensor = [x.name for x in self.weight]
        self.tensor.extend([node.output[0] for node in self.nodes if node.op_type == "Constant"])

    def get_node(self, name):
        for idx, n in enumerate(self.nodes):
            if name in n.output:
                return idx, n

    def is_tensor(self, name):
        return name in self.tensor

    def get_tensor_vale(self, name):
        for n in self.nodes:
            if name == n.output[0] and n.op_type == 'Constant':
                return onnx.numpy_helper.to_array(n.attribute[0].t)
        for w in self.weight:
            if name == w.name:
                return onnx.numpy_helper.to_array(w).astype(np.float32)

    def get_input_shape(self, name):
        for n in self.nodes:
            if name == n.output[0]:
                return self.shape_info[name]
        for w in self.weight:
            if name == w.name:
                return list(w.dims)

    def remove_cast(self):
        cast_ops = []
        flush_input = False
        for idx, node in enumerate(self.nodes):
            if node.op_type == "Cast":
                cast_ops.append(node)
                flush_input = True
                continue
            if node.op_type == "Constant":
                continue
            if flush_input:
                flush_input = False
                for i in range(len(node.input)):
                    if cast_ops[-1].output[0] == node.input[i]:
                        self.nodes[idx].input[i] = cast_ops[-1].input[0]
        for op in cast_ops:
            self.nodes.remove(op)

    def constraint(self, node, mode):
        if mode == 'broadcast' and len(node.input) == 2:
            inp_0, inp_1 = node.input
            inp_0_shape = self.get_input_shape(inp_0)
            inp_1_shape = self.get_input_shape(inp_1)
            # don't worry, in this function shape always be list
            if len(inp_0_shape) == 1 or len(inp_1_shape) == 1:
                 # normal case
                if inp_0_shape[-1] == inp_1_shape[-1] \
                    or inp_0_shape[-1] == 1 or inp_1_shape[-1] == 1:
                    return True
            elif ((inp_0_shape[-2] == 1 or inp_1_shape[-2] == 1) \
                  and inp_0_shape[:-2] == inp_1_shape[:-2]):
                # for group fc
                 return True
        else:
            raise ValueError("constrain mode: {} not support now.".format(mode))
        return False

    def normal_check(self, pninp, ninp):
        # check one node
        tensor_attr = {}
        const_tensor_remove = []
        outside_input = []
        if len(pninp) != len(ninp):
            return False, outside_input, tensor_attr, const_tensor_remove
        for p, n in zip(pninp, ninp):
            # Fold p defined as:
            # Activation: 1. (pre_pnode.output, idx), if pre_pnode has muti-output
            #             2. pre_pnode.output, if pre_pnode has one-output
            #             3. 'input', this will get input from ninp at same position
            #
            # Tensor:     1. ('input', 'tensor'), if you concern input type. eg. constant, weight
            #             2. otherwise 'input' is enough
            #             3. attr_key, get attr from tensor eg. 'axes', 'dims'. or just set any str
            #                if u don't care what the tensor will doing.
            if type(p) == tuple:
                i, j = p
                if type(i) == str and type(j) == str \
                   and i.lower() == 'input' and j.lower() == 'tensor':
                    if self.is_tensor(n):
                        outside_input.append(n)
                    else:
                        return False, outside_input, tensor_attr, const_tensor_remove
                elif type(i) == list and type(j) == int:
                    if i[j] != n:
                        return False, outside_input, tensor_attr, const_tensor_remove
                else:
                    raise ValueError("Wrong defination. {}".format(p))
            elif type(p) == str:
                if p.lower() == 'input':
                    outside_input.append(n)
                elif self.is_tensor(n):
                    # get attr form tensor value
                    tensor_attr[p] = float(self.get_tensor_vale(n))
                    const_tensor_remove.append(n)
                else:
                    raise ValueError("Invalid. This should set to be input or activation.")
                    # return False, outside_input, tensor_attr, const_tensor_remove
            elif type(p) == list and len(p) == 1:
                if p[0] != n:
                    return False, outside_input, tensor_attr, const_tensor_remove
            else:
                raise ValueError("Wrong input defination. {}".format(p))
        return True, outside_input, tensor_attr, const_tensor_remove

    def match_node(self, pnode, node):
        # match node, determin the inputs out of pattern, get attr form tensor, record the nodes which should be removed
        # check and process node's input
        matched,  outside_inp, attr, rm_tensor = self.normal_check(pnode.input, node.input)
        if not matched and (node.op_type == 'Mul' or node.op_type == 'Concat' or node.op_type == 'Add'):
            # swap input, maybe concat with >3 input doesn't work
            matched, outside_inp, attr, rm_tensor = self.normal_check(pnode.input[::-1], node.input)

        # process current node
        if matched:
            # process constraint
            if pnode.constraint is not None:
                matched = self.constraint(node, pnode.constraint)
            rm_tensor.append(node.output[0])
            # get extra attr from node's attr
            for km in pnode.attr_key_map:

                if type(km) == tuple:
                    raw_node_attr = km[0]
                    new_node_attr = km[-1]
                else:
                    raw_node_attr, new_node_attr = km, km
                if new_node_attr in attr:
                    raise ValueError("Duplicate attr name please check.")
                attr.update({new_node_attr: get_attr(node.attribute, raw_node_attr)})
        return matched,  outside_inp, attr, rm_tensor

    def match_pattern(self, pattern):
        redundancies_ops_list = []
        redundancies_ops = []
        pattern_input = []
        pattern_attr = {}
        pattern_idx = 0
        for node in self.nodes:
            match_success = False
            # for different torch version onnx graph may different.
            # eg. insert Cast, sometimes use Tensor sometimes use Weight and so on...
            if node.op_type == "Constant":
                continue
            if node.op_type == pattern[pattern_idx].op_type:
                match_success, pinp, attr, rm_op = self.match_node(pattern[pattern_idx], node)

            if match_success:
                # flush output info, so next pnode's input will be updated
                pattern[pattern_idx].output.clear()
                pattern[pattern_idx].output.extend(node.output)
                pattern_idx += 1

                pattern_attr.update(attr)
                redundancies_ops.extend(rm_op)
                # store pattern input info
                for i in pinp:
                    if i not in pattern_input:
                        pattern_input.append(i)

                if pattern_idx == len(pattern):
                    redundancies_ops_list.append(
                        RedundanciesOps(copy.copy(pattern_input), copy.copy(list(node.output)),
                                        copy.copy(pattern_attr), copy.copy(redundancies_ops)))
                    pattern_idx = 0
                    redundancies_ops.clear()
                    pattern_input.clear()
                    pattern_attr.clear()
            else:
                pattern_idx = 0
                redundancies_ops.clear()
                pattern_input.clear()
                pattern_attr.clear()
        return redundancies_ops_list

    def replace_pattern(self):
        self.remove_cast()
        replaced = False
        for op_info in self.op_list:
            ops = op_info.trg_node
            pattern = op_info.src_node
            redundancies_ops_list = self.match_pattern(pattern)
            if len(redundancies_ops_list) > 0:
                replaced = True
            for nodes in redundancies_ops_list:  # [pattern0, patter1, ...]
                node_idx, _ = self.get_node(nodes.redundancies_ops[0])
                rm_node = [self.get_node(oname)[1] for oname in nodes.redundancies_ops]
                out = nodes.pattern_output
                for i, op in enumerate(ops):
                    attr = {}
                    _input = []
                    _output = []
                    prefix = "replace_{}_{}".format(node_idx, op.op_type)
                    # get attr
                    for k in op.attr_key_map:
                        if type(k) == tuple:
                            if len(k) > 2:
                                raise ValueError("key must be one in replace pattern")
                            attr.update([(k[-1], nodes.attr[k[0]])])
                        elif type(k) == str:
                            attr.update([(k, nodes.attr[k])])
                        else:
                            raise ValueError("Wrong attr defination.")
                    attr.update(op.default)
                    # get input
                    for inp, idx in op.input:
                        if inp == "input":
                            _input.append(nodes.pattern_input[idx])
                        else:
                            _input.append(inp[idx])
                    # set output
                    if i == len(ops) - 1:   # output
                        _output = out
                    else:
                        op.output.clear()
                        op.output.extend([prefix])
                        _output = [prefix]
                    # form onnx node
                    if op.op_type == "Constant":
                        value = np.array(op.const_value)
                        new_node = onnx.helper.make_node("Constant", name=prefix, inputs=[], outputs=_output,
                                    value=onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                                                  value.shape, value))
                    else:
                        new_node = onnx.helper.make_node(op.op_type, name=prefix, inputs=_input,
                                                         outputs=_output, **attr)
                    self.nodes.insert(node_idx, new_node)
                    node_idx += 1
                for n in rm_node:
                    self.nodes.remove(n)
        return replaced

    def run(self, op_list):
        self.op_list = op_list
        fixed_point(self.replace_pattern)

        return self.nodes


class OnnxOpt(object):
    def __init__(self, model, batch_size):
        self.batch_size = batch_size
        self.model = copy.deepcopy(model)
        onnx.checker.check_model(self.model)
        self.const_tensors = []

    def get_inputs(self):
        initializer_names = [x.name for x in self.model.graph.initializer]
        return [ipt for ipt in self.model.graph.input if ipt.name not in initializer_names]

    def get_input_names(self):
        input_names = [ipt.name for ipt in self.get_inputs()]
        return input_names

    def generate_specific_rand_input(self, input_shapes):
        inputs = {}
        for key, shape in input_shapes.items():
            if len(shape) > 0 and (shape[0] == 0 or shape[0] == -1):
                if self.batch_size > 0:
                    shape [0] = self.batch_size
                else:
                    shape[0] = 1
            if not np.all(np.array(shape) > 0):
                raise RuntimeError("The shape of input '{}' has dynamic size '{}', "
                                   "please determine the input size when export "
                                   "onnx".format(key, shape))
            elem_type = self.get_elem_type(key)
            elem_type = self.get_np_type_from_elem_type(elem_type)
            if elem_type == np.bool :  # for mask
                inputs.update({key: np.random.randint(0, 2, shape, dtype=elem_type)})
            # elif elem_type == np.int64:
            #     inputs.update({key: np.random.randint(0, 10, size=shape, dtype=elem_type)})
            elif len(shape) == 0: # for idx
                inputs.update({key: np.array(0, dtype=elem_type)})
            else:
                inputs.update({key: np.random.rand(*shape).astype(elem_type)})
        return inputs

    def get_value_info_all(self, name):
        for v in self.model.graph.value_info:
            if v.name == name:
                return v
        for v in self.model.graph.input:
            if v.name == name:
                return v
        for v in self.model.graph.output:
            if v.name == name:
                return v
        return None

    @staticmethod
    def insert_elem(nodes, idx, element):
        nodes.extend([nodes[-1]])
        for i in reversed(range(idx + 1, len(nodes) - 1)):
            nodes[i].CopyFrom(nodes[i - 1])
        nodes[idx].CopyFrom(element)

    @staticmethod
    def get_shape_from_value_info_proto(vinfo):
        return [dim.dim_value for dim in vinfo.type.tensor_type.shape.dim]

    @staticmethod
    def get_np_type_from_elem_type(elem_type):
        types = (None, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.int32,
                np.int64, str, np.bool, np.float16, np.double, np.uint32, np.uint64,
                np.complex64, np.complex128, np.float16)
        assert len(types) == 17
        _type = types[elem_type]
        assert _type is not None
        return _type

    def get_shape(self, name):
        vinfo = self.get_value_info_all(name)
        if vinfo is None:
            raise RuntimeError("Can't get shape of '{}'".format(name))
        return self.get_shape_from_value_info_proto(vinfo)

    def get_elem_type(self, name):
        vinfo = self.get_value_info_all(name)
        if vinfo is None:
            raise RuntimeError("Can't get dtype of '{}'".format(name))
        return vinfo.type.tensor_type.elem_type

    def is_dynamic(self, node):
        if node.op_type in ["NonMaxSuppression", "NonZero", "Unique"] \
                and node.input[0] not in self.const_tensors:
            return True
        if node.op_type in ["Reshape", "Expand", "Upsample", "ConstantOfShape"] \
                and len(node.input) > 1 and node.input[1] not in self.const_tensors:
            return True
        if node.op_type in ["Resize"] \
                and ((len(node.input) > 2 and node.input[2] not in self.const_tensors) \
                    or (len(node.input) > 3 and node.input[3] not in self.const_tensors)):
            return True
        return False

    def has_subgraph_in_node(self, node):
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return True
        return False

    def is_quantizeLinear(self, node):
        return node.op_type in ["DequantizeLinear", "QuantizeLinear"]

    def is_non_determinstic_node(self, node):
        return node.op_type in ["RandomNormal", "RandomNormalLike", "RandomUniformLike"]

    def get_constant_nodes(self):
        const_nodes = []
        dynamic_tensors = []
        self.const_tensors = [x.name for x in self.model.graph.initializer]
        self.const_tensors.extend([node.output[0] for node in self.model.graph.node if node.op_type == "Constant"])
        for node in self.model.graph.node:
            if any(x in dynamic_tensors for x in node.input):
                dynamic_tensors.extend(node.output)
            elif node.op_type == "Shape":
                const_nodes.append(node)
                self.const_tensors.extend(node.output)
            elif self.is_dynamic(node):
                dynamic_tensors.extend(node.output)
            elif self.is_quantizeLinear(node):
                pass
            elif self.has_subgraph_in_node(node):
                pass
            elif len(node.input) > 0 and all([x in self.const_tensors for x in node.input]) \
                    and not self.is_non_determinstic_node(node):
                const_nodes.append(node)
                self.const_tensors.extend(node.output)
        return copy.deepcopy(const_nodes)

    def forward(self, model):
        input_shapes = {}
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel(0)
        sess_options.log_severity_level = 3

        sess = rt.InferenceSession(model.SerializeToString(), sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])
        input_names = self.get_input_names()
        inputs = {}
        for name in input_names:
            shape = self.get_shape(name)
            input_shapes.update({name: shape})
        inputs.update(self.generate_specific_rand_input(input_shapes))
        outputs = [x.name for x in sess.get_outputs()]
        run_options = rt.RunOptions()
        run_options.log_severity_level = 3
        return OrderedDict(zip(outputs, sess.run(outputs, inputs, run_options=run_options)))

    def forward_for_node_outputs(self, const_nodes):
        model = copy.deepcopy(self.model)
        for node in const_nodes:
            for output in node.output:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        return self.forward(model)

    def eliminate_const_nodes(self, const_node, res):
        do_eliminate = False
        for i, node in enumerate(self.model.graph.node):
            if node in const_node:
                for output in node.output:
                    new_node = copy.deepcopy(node)
                    new_node.name = "node_" + output
                    new_node.op_type = "Constant"
                    new_attr = onnx.helper.make_attribute(
                        "value",
                        onnx.numpy_helper.from_array(res[output], name=output)
                    )
                    del new_node.input[:]
                    del new_node.attribute[:]
                    del new_node.output[:]
                    new_node.output.extend([output])
                    new_node.attribute.extend([new_attr])
                    self.insert_elem(self.model.graph.node, i + 1, new_node)
                del self.model.graph.node[i]
                do_eliminate = True
        return do_eliminate

    def remove_unused_nodes(self):
        node_inputs = []
        unused_node = []
        for n in self.model.graph.node:
            node_inputs.extend(n.input)
        node_inputs.extend([out.name for out in self.model.graph.output])
        node_inputs = set(node_inputs)

        for n in self.model.graph.node:
            if len(set(n.output).intersection(node_inputs)) == 0:
                unused_node.append(n)
        for n in unused_node:
            self.model.graph.node.remove(n)

    def infer_shapes(self):
        try:
            self.model = onnx.shape_inference.infer_shapes(self.model)
        except:
            pass

    def constant_folding(self, infer_shapes=True):
        const_nodes = self.get_constant_nodes()
        res = self.forward_for_node_outputs(const_nodes)
        const_node = [node for node in const_nodes if node.output[0] in res]
        do_eliminate = self.eliminate_const_nodes(const_node, res)
        onnx.checker.check_model(self.model)
        if infer_shapes:
            self.infer_shapes()
        return do_eliminate

    def run(self, dump):
        fixed_point(self.constant_folding)
        self.remove_unused_nodes()
        if dump:
            dump_model(self.model, "constant_opt.onnx")
        return self.model

def onnx_opt(model, batch_size, dump=False):
    constant_opt = OnnxOpt(model, batch_size)
    model = constant_opt.run(dump)
    fdef = Form_Deform(model)

    # torch.not_equal
    eq_not_op0 = PesudoNode("Equal", ["input", "input"])
    eq_not_op1 = PesudoNode("Not", [eq_not_op0.output,])
    eq_not_op = PesudoNode("Equal", [("input", 0), ("input", 1)], default={"not": True})
    eq_not = FoldUnfoldInfo([eq_not_op0, eq_not_op1], [eq_not_op])

    # torch.std
    std_ub_op0 = PesudoNode("ReduceMean", ["input",], attr_key_map=[("axes", "dim"),])
    std_ub_op1 = PesudoNode("Sub", ["input", std_ub_op0.output])
    std_ub_op2 = PesudoNode("Mul", [std_ub_op1.output, std_ub_op1.output])
    std_ub_op3 = PesudoNode("ReduceMean", [std_ub_op2.output,], attr_key_map=["keepdims",])
    std_ub_op5 = PesudoNode("Mul", [std_ub_op3.output, "dont_care"])
    std_ub_op7 = PesudoNode("Div", [std_ub_op5.output, "dont_care"])
    std_ub_op8 = PesudoNode("Sqrt", [std_ub_op7.output,])
    std_ub_op = PesudoNode("Std", [("input", 0),], attr_key_map=["dim", "keepdims",], default={"unbiased": True})
    std_ub = FoldUnfoldInfo([std_ub_op0, std_ub_op1, std_ub_op2, std_ub_op3,
                             std_ub_op5, std_ub_op7, std_ub_op8], [std_ub_op])

    std_op0 = PesudoNode("ReduceMean", ["input",], attr_key_map=[("axes", "dim"),])
    std_op1 = PesudoNode("Sub", ["input", std_op0.output])
    std_op2 = PesudoNode("Mul", [std_op1.output, std_op1.output])
    std_op3 = PesudoNode("ReduceMean", [std_op2.output,], attr_key_map=["keepdims",])
    std_op4 = PesudoNode("Sqrt", [std_op3.output,])
    std_op = PesudoNode("Std", [("input", 0),], attr_key_map=["dim", "keepdims",], default={"unbiased": False})
    std = FoldUnfoldInfo([std_op0, std_op1, std_op2, std_op3, std_op4], [std_op])

    # torch.Where
    where_op = PesudoNode("Where", ["input", "input", "input"])
    where_op0 = PesudoNode("Mul", [("input", 0), ("input", 1)])          # mask * cond
    where_op1 = PesudoNode("Constant", const_value=[-1])
    where_op2 = PesudoNode("Constant", const_value=[1])
    where_op3 = PesudoNode("Mul", [("input", 0), (where_op1.output, 0)])           # mask * -1
    where_op4 = PesudoNode("Add", [(where_op3.output, 0), (where_op2.output, 0)])  # -mask + 1
    where_op5 = PesudoNode("Mul", [("input", 2), (where_op4.output, 0)])           # y * (-mask + 1)
    where_op6 = PesudoNode("Add", [(where_op0.output, 0), (where_op5.output, 0)])  # -mask + 1
    where = FoldUnfoldInfo([where_op], [where_op0, where_op1, where_op2, where_op3, where_op4, where_op5, where_op6])

    # torch.LayerNorm
    layernorm_aff_op0 = PesudoNode("ReduceMean", ["input",], attr_key_map=[("axes",)])
    layernorm_aff_op1 = PesudoNode("Sub", ["input", layernorm_aff_op0.output])
    layernorm_aff_op3 = PesudoNode("Pow", [layernorm_aff_op1.output, "dont_care"])
    layernorm_aff_op4 = PesudoNode("ReduceMean", [layernorm_aff_op3.output,])
    layernorm_aff_op6 = PesudoNode("Add", [layernorm_aff_op4.output, "eps"])
    layernorm_aff_op7 = PesudoNode("Sqrt", [layernorm_aff_op6.output,])
    layernorm_aff_op8 = PesudoNode("Div", [layernorm_aff_op1.output, layernorm_aff_op7.output])
    layernorm_aff_op9 = PesudoNode("Mul", [layernorm_aff_op8.output, "input"])
    layernorm_aff_op10 = PesudoNode("Add", [layernorm_aff_op9.output, "input"])
    layernorm_aff_op = PesudoNode("LayerNorm", [("input", 0), ("input", 1), ("input", 2)],
                                  attr_key_map=["axes", "eps",], default={"elementwise_affine": True})
    layernorm_aff = FoldUnfoldInfo([layernorm_aff_op0, layernorm_aff_op1,
                                    layernorm_aff_op3, layernorm_aff_op4,
                                    layernorm_aff_op6, layernorm_aff_op7, layernorm_aff_op8,
                                    layernorm_aff_op9, layernorm_aff_op10], [layernorm_aff_op])

    layernorm_op0 = PesudoNode("ReduceMean", ["input",], attr_key_map=[("axes",)])
    layernorm_op1 = PesudoNode("Sub", ["input", layernorm_op0.output])
    layernorm_op3 = PesudoNode("Pow", [layernorm_op1.output, "dont_care"])
    layernorm_op4 = PesudoNode("ReduceMean", [layernorm_op3.output, ])
    layernorm_op6 = PesudoNode("Add", [layernorm_op4.output, "eps"])
    layernorm_op7 = PesudoNode("Sqrt", [layernorm_op6.output, ])
    layernorm_op8 = PesudoNode("Div", [layernorm_op1.output, layernorm_op7.output])
    layernorm_op = PesudoNode("LayerNorm", [("input", 0),],
                                  attr_key_map=["axes", "eps", ], default={"elementwise_affine": False})
    layernorm = FoldUnfoldInfo([layernorm_op0, layernorm_op1, layernorm_op3, layernorm_op4,
                                    layernorm_op6, layernorm_op7, layernorm_op8], [layernorm_op])

    # matmul + bias
    matmul_bias_op0 = PesudoNode("MatMul", ["input", ("input", "tensor")])
    matmul_bias_op1 = PesudoNode("Add", [matmul_bias_op0.output, ("input", "tensor")], constraint="broadcast")
    matmul_bias_op = PesudoNode("MatMul", [("input", 0), ("input", 1), ("input", 2)],)
    matmul_bias = FoldUnfoldInfo([matmul_bias_op0, matmul_bias_op1], [matmul_bias_op])

    fdef.run([eq_not, std_ub, std, where, layernorm_aff, layernorm, matmul_bias])

    if dump:
        dump_model(model, "final_opt.onnx")
    return model
