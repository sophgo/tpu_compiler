from transform.mlirimporter import MLIRImporter

import numpy as np

def calcConv2DSpatial(i, kernel, stride, padding, dilation):
    return (i + 2*padding - dilation * (kernel - 1) - 1)/stride + 1

mlir_path = "test_conv.mlir"
weight_npz = "weight_1111.npz"
input_npz = "test_in_fp32.npz"

output_name = "conv_output"
output_golden_npz = "test_output_golden.npz"

if __name__ == "__main__":
    conv_param = {
        'stride_h':  1,
        'stride_w':  1,
        'padding': "VALID",
        'dilation_h': 1,
        'dilation_w': 1,
        'group': 1,
        'is_dw': False,
        'with_bias': True,
        'do_relu': False,
    }

    input_shape = [1, 3, 224, 224]
    filter_shape = [64, 3, 3, 3]
    bias_shape = [64]
    oh = calcConv2DSpatial(input_shape[2], filter_shape[2], conv_param['stride_h'], 0, conv_param['dilation_h'])
    ow = calcConv2DSpatial(input_shape[3], filter_shape[3], conv_param['stride_w'], 0, conv_param['dilation_w'])
    output_shape = [1, filter_shape[0], oh, ow]

    importer = MLIRImporter([input_shape], [output_shape])
    input = importer.add_input_op("input", 0) # mlir first args

    # add input npz
    input_data = np.ones(tuple(input_shape), dtype=np.float32)
    np.savez(input_npz, **{'input': input_data})

    # add Weight file name, filter
    importer.add_weight_file_op(weight_npz)
    # filter
    filter_op = importer.add_load_file_op("fileter", filter_shape)
    filter_data = np.ones(tuple(filter_shape), dtype=np.float32)

    # bias
    bias_op = importer.add_load_file_op("bias", bias_shape)
    bias_data = np.ones(tuple(bias_shape), dtype=np.float32)

    np.savez(weight_npz, **{"fileter": filter_data, "bias": bias_data})

    # MLIR add conv op
    conv_op = importer.add_conv_op(output_name, [input, filter_op, bias_op], output_shape, **conv_param)
    importer.add_return_op([conv_op]) # return is list

    # Gen mlir
    with open(mlir_path, "w") as f:
        f.write(importer.print_module())

    # Create output golden
    # 3*3 kernel all one 3 channel => 3*3*3 = 27
    # bias = 1, output=27+1=28
    golden_data = 28 * np.ones(tuple(output_shape), dtype=np.float32)
    np.savez(output_golden_npz, **{output_name: golden_data})
