from transform.mlirimporter import BaseConverterInterface, MLIRImporter, checkKey
from onnx import numpy_helper, mapping
from termcolor import colored, cprint
from math import floor, ceil
from numbers import Number

import logging
import numpy as np

def calcConv2DSpatial(i, kernel, stride, padding, dilation):
    #[i + 2*p - k - (k-1)*(d-1)]/s + 1
    return (i + 2*padding - dilation * (kernel - 1) - 1)/stride + 1
def calcPool2DFloor(i, kernel, stride, padding):
    return floor((i + 2 * padding - kernel) / stride) + 1

def calcPool2DCeil(i, kernel, stride, padding):
    return ceil((i + 2 * padding - kernel) / stride) + 1

def get_shape_size(shape):
    size = 1
    for i in shape:
        size*=i
    return size

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
            "Add": lambda node: self.convert_add_op(node),
            "BatchNormalization": lambda node: self.convert_batchnorm_op(node),
            "Concat": lambda node: self.convert_concat_op(node),
            "Conv": lambda node: self.convert_conv_op(node),
            "Constant": lambda node: self.convert_constant_op(node),
            "Flatten": lambda node: self.convert_flatten_op(node),
            "Gather": lambda node: self.convert_gather_op(node),
            "Gemm": lambda node: self.convert_gemm_op(node),
            "GlobalAveragePool": lambda node: self.convert_global_avg_pool_op(node),
            "MaxPool": lambda node: self.convert_maxpool_op(node),
            "Mul" : lambda node: self.convert_mul_op(node),
            "Relu": lambda node: self.convert_relu_op(node),
            "Reshape": lambda node: self.convert_reshape_op(node),
            "Shape": lambda node: self.convert_shape_op(node),
            "Sigmoid" :lambda node: self.convert_sigmoid_op(node),
            "Transpose": lambda node: self.convert_transpose_op(node),
            "Unsqueeze": lambda node: self.convert_unsqueeze_op(node),
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
            for dim in output.type.tensor_type.shape.dim:
                output_shape.append(dim.dim_value)
            outputs.append(output_shape)
        
        # init importer
        self.CVI = MLIRImporter(inputs, outputs)
    
    def addOperand(self, op_name, op, shape):
        self.valueMap[op_name] = (op, shape)
    
    def getOperand(self, op_name):
        return self.valueMap[op_name]
    
    def getTensor(self, op_name):
        find_tensor = [t for t in self.converted_tensors if t.name == op_name]
        if len(find_tensor) < 1:
            raise RuntimeError("No {} tensor in model".format(op_name))
        else:
            return find_tensor[0]
    
    def TensortoNpz(self):
        tensor_npz = {}
        for i in self.converted_tensors:
            tensor_npz[i.name] = i.tensor_data
        np.savez(self.output_tensor_file, **tensor_npz)

    @staticmethod
    def unsqueeze_shape(shape, axis):
        new_shape = [n for n in shape]
        for n in axis:
            new_shape.insert(n, 1)
        return new_shape
        
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

        # add return op
        return_op = list()
        # Set output
        for output in self.output_nodes:
            op, _ = self.getOperand(output.name)
            return_op.append(op)

        self.CVI.add_return_op(return_op)
        mlir_txt = self.CVI.print_module()
        with open("resnet50.mlir", "w") as f:
            f.write(mlir_txt)

    
    def convert_add_op(self, onnx_node):
        assert(len(onnx_node.inputs) == 2)
        op1, input_shape1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2 = self.getOperand(onnx_node.inputs[1])

        if input_shape1 != input_shape2:
            raise AttributeError("{} v.s. {} shape not same".format(input_shape1, input_shape2)) 
        operands = list()
        operands.append(op1)
        operands.append(op2)
        output_shape = input_shape1

        add_op = self.CVI.add_eltwise_add_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, add_op, output_shape)

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
    
    def convert_constant_op(self, onnx_node):
        """
            Constant Op is tensor data at IR, 
            we change it to load weight tensor, and store
        """
        assert(onnx_node.op_type == "Constant")
        onnx_tensor = onnx_node.attrs['value']
        np_tensor =  numpy_helper.to_array(onnx_tensor)
        data_type = onnx_dtype(onnx_tensor.data_type)
  
        if data_type in [np.float32, np.float64, np.int32, np.int64]:
            np_tensor = np_tensor.astype(np.float32).flatten()
            # add new weight tensor
            new_tensor = OnnxTensor(onnx_node.name, np_tensor, np_tensor.shape)
            constant_op = self.CVI.add_load_file_op(onnx_node.name, np_tensor.shape)
            self.converted_tensors.append(new_tensor)
            self.addOperand(onnx_node.name, constant_op, np_tensor.shape)

        else:
            raise ValueError("Not Support {} type".format(data_type))

    def convert_concat_op(self, onnx_node):
        assert(onnx_node.op_type == "Concat")
        if len(onnx_node.inputs) < 2:
            raise ValueError("{} must great than 2".format(onnx_node.op_type))
        op1, input_shape1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2 = self.getOperand(onnx_node.inputs[1])

        axis = onnx_node.attrs['axis']
        operands = [op1, op2]
        output_shape = list()
        for idx, (s1, s2) in enumerate(zip(input_shape1, input_shape2)):
            if  idx== axis:
                output_shape.append(s1+s2)
            else:
                assert(s1 == s2)
                output_shape.append(s1)

        concat_op = self.CVI.add_concat_op(onnx_node.name, operands, output_shape, axis=axis)
        self.addOperand(onnx_node.name, concat_op, output_shape)

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
        operands = list()
        operands.append(op)
        for weight_name in onnx_node.inputs[1:]:
            tensor = self.getTensor(weight_name)
            if tensor == None:
                raise RuntimeError("No {} tensor in model".format(weight_name))
            weight_op = self.CVI.add_load_file_op(tensor.name, tensor.shape)
            operands.append(weight_op)
            
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
        conv_op = self.CVI.add_conv_op(onnx_node.name, operands, output_shape, **conv_param)
        self.addOperand(onnx_node.name, conv_op, output_shape)

    def convert_flatten_op(self, onnx_node):
        assert(onnx_node.op_type == "Flatten")
        if onnx_node.attrs["axis"] != 1:
            raise AttributeError("TODO: axis != 1 case")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = [input_shape[0], input_shape[1]]
        reshape_op = self.CVI.add_reshape_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, reshape_op, output_shape)

    def convert_gather_op(self, onnx_node):
        """
            first input is tensor data, second input is constant
        """
        assert(onnx_node.op_type == "Gather")
        op, input_shape = self.getOperand(onnx_node.inputs[0])

        if 'axis' in onnx_node.attrs:
            axis = onnx_node.attrs['axis']
        else:
            axis = 0
       
        gather_indices = self.getTensor(onnx_node.inputs[1]).tensor_data
        new_shape = input_shape
        if new_shape[axis] > len(gather_indices):
            new_shape[axis] = len(gather_indices)
        else:
            raise ValueError("Gather input shape dim {} ({}) must great than {} ({})".format(axis, input_shape, len(gather_indices), gather_indices))
        # TODO: our IR no Gather function, please add
        # Hardcode Here
        self.addOperand(onnx_node.name, op, new_shape)

    def convert_gemm_op(self, onnx_node):
        assert(onnx_node.op_type == "Gemm")
        #(M, K) * (K, N) => (M, N)
        op, input_shape = self.getOperand(onnx_node.inputs[0])

        operands = list()
        operands.append(op)
        weight_name = onnx_node.inputs[1]
        weight_tensor = self.getTensor(weight_name)
        weight_op = self.CVI.add_load_file_op(weight_name, weight_tensor.shape)
        operands.append(weight_op)

        bias_name = onnx_node.inputs[2]
        bias_tensor = self.getTensor(bias_name)
        bias_op = self.CVI.add_load_file_op(bias_name, bias_tensor.shape)
        operands.append(bias_op)

        M = input_shape[0]
        K = input_shape[1]
        N = bias_tensor.shape[0]
        output_shape = [M, N]
        fc_op = self.CVI.add_fully_connected_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, fc_op, output_shape)

    def convert_global_avg_pool_op(self, onnx_node):
        assert(onnx_node.op_type == "GlobalAveragePool")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        # print(input_shape)
        on = input_shape[0]
        oc = input_shape[1]
        
        pool_avg_2d_param = {
            'stride_h':  1,
            'stride_w':  1,
            'kernel_h':  input_shape[2],
            'kernel_w':  input_shape[3],
            'padding_b': 0,
            'padding_r': 0,
            'padding_t': 0,
            'padding_l': 0,
            'do_relu': False,
        }
        output_shape = [int(on), int(oc), 1, 1]
        pool_avg_op = self.CVI.add_pool_avg_2d_op(onnx_node.name, operands, output_shape, **pool_avg_2d_param)
        self.addOperand(onnx_node.name, pool_avg_op, output_shape)
    
    def convert_maxpool_op(self, onnx_node):
        assert(onnx_node.op_type == "MaxPool")
        pool_max_2d_param = {
            'stride_h': onnx_node.attrs['strides'][0], 
            'stride_w': onnx_node.attrs['strides'][1], 
            'kernel_h': onnx_node.attrs['kernel_shape'][0],
            'kernel_w': onnx_node.attrs['kernel_shape'][1],
            'padding_b': onnx_node.attrs['pads'][0],
            'padding_r': onnx_node.attrs['pads'][1],
            'padding_t': onnx_node.attrs['pads'][2],
            'padding_l': onnx_node.attrs['pads'][3],
            'do_relu': False,
        }
        
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        on = input_shape[0]
        oc = input_shape[1]
        oh = calcPool2DFloor(input_shape[2], onnx_node.attrs['kernel_shape'][0], onnx_node.attrs['strides'][0], onnx_node.attrs['pads'][0])
        ow = calcPool2DFloor(input_shape[3], onnx_node.attrs['kernel_shape'][1], onnx_node.attrs['strides'][1], onnx_node.attrs['pads'][1])
        output_shape = [int(on), int(oc), int(oh), int(ow)]
        pool_max_op = self.CVI.add_pool_max_2d_op(onnx_node.name, operands, output_shape, **pool_max_2d_param)
        self.addOperand(onnx_node.name, pool_max_op, output_shape)

    def convert_relu_op(self, onnx_node):
        assert(onnx_node.op_type == "Relu")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        operands = list()
        operands.append(op)
        output_shape = input_shape
        relu_op = self.CVI.add_relu_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, relu_op, output_shape)

    def convert_mul_op(self, onnx_node):
        assert(onnx_node.op_type == "Mul")  
        op1, input_shape1 = self.getOperand(onnx_node.inputs[0])
        op2, input_shape2 = self.getOperand(onnx_node.inputs[1])
        operands = list()
        operands.append(op1)
        operands.append(op2)
        axis = 0
        for idx, (d1, d2) in enumerate(zip(input_shape1, input_shape2)):
            if d1 != d2:
              axis = idx - 1
        

        output_shape = input_shape1
        broadcast_mul_op = self.CVI.add_broadcast_mul_op(onnx_node.name, operands, output_shape, axis=axis)
        self.addOperand(onnx_node.name, broadcast_mul_op, output_shape)

    def convert_reshape_op(self, onnx_node):
        assert(onnx_node.op_type == "Reshape")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        _ , new_shape = self.getOperand(onnx_node.inputs[1])
        if get_shape_size(input_shape) != get_shape_size(new_shape):
            raise ValueError("can't reshape {} to {}, size different".format(input_shape, new_shape))
        operands = list()
        operands.append(op)
        output_shape = new_shape
        relu_op = self.CVI.add_relu_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, relu_op, output_shape)

    def convert_shape_op(self, onnx_node):
        assert(onnx_node.op_type == "Shape")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        data = np.array(input_shape)
        new_tensor = OnnxTensor(onnx_node.name, data, list(data.shape))
        self.converted_tensors.append(new_tensor)
        shape_op = self.CVI.add_load_file_op(onnx_node.name, new_tensor.shape)
        self.addOperand(onnx_node.name, shape_op, list(data.shape))

    def convert_sigmoid_op(self, onnx_node):
        assert(onnx_node.op_type == "Sigmoid")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        operands = [op]
        output_shape = input_shape
        sigmoid_op = self.CVI.add_sigmoid_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, sigmoid_op, output_shape)

    def convert_unsqueeze_op(self, onnx_node):
        """Unsqueeze """
        assert(onnx_node.op_type == "Unsqueeze")
        op, input_shape = self.getOperand(onnx_node.inputs[0])
        checkKey(onnx_node.attrs, 'axes')
        operands = [op]
        axis_value_list = onnx_node.attrs['axes']
        output_shape = self.unsqueeze_shape(input_shape, axis_value_list)
        reshape_op = self.CVI.add_reshape_op(onnx_node.name, operands, output_shape)
        self.addOperand(onnx_node.name, reshape_op, output_shape)


    def run(self):
        self.convert_node()
        self.convert_tensor()
        self.convert_graph()
        self.TensortoNpz()

