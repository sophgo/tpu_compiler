from transform.mlirimporter import BaseConverterInterface, MLIRImporter
from onnx import numpy_helper
from termcolor import colored, cprint
from math import floor, ceil

import logging
import numpy as np

def calcConv2DSpatial(i, kernel, stride, padding, dilation):
    #[i + 2*p - k - (k-1)*(d-1)]/s + 1
    return (i + 2*padding - dilation * (kernel - 1) - 1)/stride + 1
def calcPool2DFloor(i, kernel, stride, padding):
    return floor((i + 2 * padding - kernel) / stride) + 1

def calcPool2DCeil(i, kernel, stride, padding):
    return ceil((i + 2 * padding - kernel) / stride) + 1

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
}

def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)

def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

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
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

class OnnxNode():
    def __init__(self, node):
        self.name = str(node.name)
        if self.name == '':
            self.name = str(node.output[0])
        self.op_type = str(node.op_type)
        self.attrs = dict([(attr.name, translate_onnx(attr.name, convert_onnx_attribute_proto(attr))) for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def print_info(self):
        cprint("node: {}".format(self.name), 'cyan')
        cprint("    type: {}".format(self.op_type), 'white')
        cprint("    inputs: {}".format(self.inputs), 'white')
        cprint("    outputs: {}".format(self.outputs), 'white')
        cprint("    attrs: {}".format(self.attrs), 'white')
        for arg in self.attrs:
            cprint("        {}: {}".format(arg, self.attrs[arg]), 'green')

class OnnxTensor():
    def __init__(self, name, value, shape):
        self.name = name
        self.tensor_data = value
        self.shape = shape
    
    def print_info(self):
        cprint("tensor: {}".format(self.name), 'cyan')
        cprint("    shape: {}".format(self.shape), 'white')
       


class OnnxConverter(BaseConverterInterface):
    def __init__(self, model_name, onnx_model):
        self.input_nodes = onnx_model.graph.input
        self.output_nodes = onnx_model.graph.output
        self.nodes = onnx_model.graph.node
        self.tensors = onnx_model.graph.initializer

        self.converted_nodes = list()
        self.converted_tensors = list()

        self.valueMap = dict() # {op_name: (mlir op, shape)}
        self.CVI = None
        self.init_importer()
        self.output_tensor_file = "{}_1_06eeeb7e.npz".format(model_name)
        self.onnxop_factory = {
            "Conv": lambda node: self.convert_conv_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node)
        }
    def init_importer(self):
        # get input shape
        inputs = list()
        for input in self.input_nodes:
            input_shape = list()
            for dim in input.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value)
            inputs.append(input_shape)
        # get output shape
        outputs = list()
        for output in self.output_nodes:
            output_shape = list()
            for dim in input.type.tensor_type.shape.dim:
                output_shape.append(dim.dim_value)
            outputs.append(input_shape)
        
        # init importer
        self.CVI = MLIRImporter(inputs, outputs)
    
    def addOperand(self, op_name, op, shape):
        self.valueMap[op_name] = (op, shape)
    
    def getOperand(self, op_name):
        return self.valueMap[op_name]
    
    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            return None
        else:
            return find_tensor[0]
    
    def convert_node(self):
        """convert onnx node to OnnxNode"""
        for n in self.nodes:
            node = OnnxNode(n)
            node.print_info()
            self.converted_nodes.append(node)
    
    def convert_tensor(self):
        """convert onnx tensor to OnnxTensor"""
        for tensor in self.tensors:
            name = tensor.name
            shape = list(tensor.dims)
            data = numpy_helper.to_array(tensor).astype(np.float32)
            tensor = OnnxTensor(name, data, shape)
            #tensor.print_info()
            self.converted_tensors.append(tensor)
    
    def convert_graph(self):
        """convert all to mlir"""
        # add weight op
        self.CVI.add_weight_file_op(self.output_tensor_file)

        # add input op 
        for idx, input in enumerate(self.input_nodes):
            input_shape = list()
            for dim in input.type.tensor_type.shape.dim:
                input_shape.append(dim.dim_value)
            input_op = self.CVI.add_input_op(input.name, idx)
            self.addOperand(input.name, input_op, input_shape)
        def NoneAndRaise(node):
            raise RuntimeError("{} Op not support now".format(node.op_type))
        # add node op
        for n in self.converted_nodes:
            n.print_info()
            self.onnxop_factory.get(n.op_type, lambda x: NoneAndRaise(x))(n)        
            self.CVI.print_module()
        # add return op


    def convert_conv_op(self, onnx_node):
        assert(onnx_node.op_type == "Conv")
        conv_param = {
            'stride_h':  onnx_node.attrs['strides'][0],
            'stride_w':  onnx_node.attrs['strides'][1],
            'padding': "SAME" if onnx_node.attrs['pads'][0] > 0 else "VALID",
            'dilation_h': onnx_node.attrs['dilations'][0],
            'dilation_w': onnx_node.attrs['dilations'][1],
            'group': onnx_node.attrs['group'],
            'is_dw': False,
            'with_bias': len(onnx_node.inputs) > 2,
            'do_relu': False,
        }
        op, shape = self.getOperand(onnx_node.inputs[0])
        opreands = list()
        for weight_name in onnx_node.inputs[1:]:
            tensor = self.getTensor(weight_name)
            if tensor == None:
                raise RuntimeError("No {} tensor in model".format(weight_name))
            weight_op = self.CVI.add_load_file_op(tensor.name, tensor.shape)
            opreands.append(weight_op)
            
        on = shape[0]
        oc = tensor.shape[0] # feature map size
        oh = calcConv2DSpatial(
            shape[2], 
            onnx_node.attrs['kernel_shape'][0], 
            onnx_node.attrs['strides'][0], 
            onnx_node.attrs['pads'][0], 
            onnx_node.attrs['dilations'][0]
        )
        ow = calcConv2DSpatial(
            shape[3], 
            onnx_node.attrs['kernel_shape'][1], 
            onnx_node.attrs['strides'][1], 
            onnx_node.attrs['pads'][1], 
            onnx_node.attrs['dilations'][1]
        )
        output_shape = [on, oc, oh, ow]
        conv_op = self.CVI.add_conv_op(onnx_node.name, opreands, output_shape, **conv_param)
        self.addOperand(onnx_node.name, conv_op, output_shape)

    def convert_batchnorm_op(self, onnx_node):
        assert(onnx_node.op_type == "BatchNormalization")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        epsilon = onnx_node.attrs['epsilon']
        # we fuse batchnorm and scale at here
        gamma_value = self.getTensor(onnx_node.inputs[1]).tensor_data
        beta_value = self.getTensor(onnx_node.inputs[2]).tensor_data
        mean_value = self.getTensor(onnx_node.inputs[3]).tensor_data
        var_value = self.getTensor(onnx_node.inputs[4]).tensor_data
        
        scale_name = "{}_0".format(onnx_node.name)
        scale_value = ((1.0 / np.sqrt(
                    var_value + epsilon)) * gamma_value)
        
        scale_op = self.CVI.add_load_file_op(scale_name, self.getTensor(onnx_node.inputs[1]).shape)
        # add new weight tensor
        new_tensor_1 = OnnxTensor(scale_name, scale_value, self.getTensor(onnx_node.inputs[1]).shape)
        self.converted_tensors.append(new_tensor_1)

        offset_name =  "{}_1".format(onnx_node.name)
        offset_value = (-mean_value * scale_value) + beta_value
        offset_op = self.CVI.add_load_file_op(offset_name, self.getTensor(onnx_node.inputs[1]).shape)
        # add new bias tensor
        new_tensor_2 = OnnxTensor(offset_name, offset_value, self.getTensor(onnx_node.inputs[1]).shape)
        self.converted_tensors.append(new_tensor_2)


        operands.append(scale_op)
        operands.append(offset_op)

        output_shape = input_shape
        scaleop = self.CVI.add_scale_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, scaleop, output_shape)

    def run(self):
        self.convert_node()
        self.convert_tensor()
        self.convert_graph()

