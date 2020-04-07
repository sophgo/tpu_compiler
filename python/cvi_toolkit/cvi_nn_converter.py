#!/usr/bin/python3
import sys, re, os
import yaml
import argparse
import site

if os.environ.get('LD_LIBRARY_PATH', None) is None:
    os.environ['LD_LIBRARY_PATH'] = "{}/lib".format(sys.prefix) + ':' + "{}/lib".format(site.USER_BASE)
    print("Setting LD_LIBRARY_PATH {}".format(os.environ['LD_LIBRARY_PATH']))
else:
    if not sys.prefix in os.environ['LD_LIBRARY_PATH'] or not site.USER_BASE in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ':' + "{}/lib".format(sys.prefix) + ':' + "{}/lib".format(site.USER_BASE)

        try:
            os.execv(sys.argv[0], sys.argv)
        except Exception as e:
            sys.exit('EXCEPTION: Failed to Execute under modified environment, ' + e)


from cvi_toolkit import cvinn, preprocess
from cvi_toolkit.numpy_helper import npz_extract, npz_rename
from cvi_toolkit import cvi_data
net = cvinn()
cvi_data_tool = cvi_data()
preprocessor = preprocess()



def parse(config: dict):
    # model to mlir
    model_name = None
    output_file = config.get("output_file", None)
    model_name = output_file.split('.')[0].split('/')[-1]
    Convert_model = config.get('Convert_model', None)
    if Convert_model:
        t = Convert_model
        model_type = t.get('framework_type')
        model_file = t.get('model_file')
        weight_file = t.get('weight_file', None)
        tpu_op_info = "{}_op_info.csv".format(model_name)
        fp32_mlirfile = "{}.mlir".format(model_name)
        try:
            net.convert_model(model_type, model_file, fp32_mlirfile, weight_file=weight_file, tpu_op_info=tpu_op_info)
        except RuntimeError as err:
            print("RuntimeError {}".format(err))
            exit(-1)

    else:
        print('No Convert_model in yml')
        exit(-1)

    # preprocess input data and save input_npz
    data_preprocess = config.get("data_preprocess", None)
    fp32_in_npz = None
    if data_preprocess:
        t = data_preprocess
        mean = t.get('image_mean', None)
        channel_mean = t.get('channel_mean', None)
        mean_file = t.get('mean_file', None)

        if mean != None and channel_mean != None:
            print("[Error]channel_mean and mean should not be set at the same time!")
            exit(-1)
        elif channel_mean != None:
            mean = channel_mean

        if mean != None and mean_file != None:
            print("[Error]mean_file and mean value should not be set at the same time!")
            exit(-1)

        input_scale = t.get('input_scale', 1.0)
        net_input_dims = t.get('net_input_dims')
        if net_input_dims == None :
            print("[Error]net_input_dims should be set in yml file !!")
            exit(-1)

        raw_scale = t.get('raw_scale', 255.0)
        transpose = t.get('transpose')
        resize_dims = t.get('image_resize_dim')
        letter_box = t.get('LetterBox', False)

        rgb_order = t.get('RGB_order',"bgr")
        npz_input = t.get('npz_input')

        input_file = t.get('input_file')
        if input_file == None :
            print('[Error] Please set input file image in yml!')
            exit(-1)

        output_npz = t['output_npz']
        fp32_in_npz = output_npz
        preprocessor.config(net_input_dims=net_input_dims,
                    resize_dims=resize_dims,
                    mean=mean,
                    mean_file=mean_file,
                    input_scale=input_scale,
                    raw_scale=raw_scale,
                    transpose=transpose,
                    rgb_order=rgb_order,
                    npz_input=npz_input,
                    letter_box=letter_box)

        ret = preprocessor.run(input_file, output_npz=output_npz)
        if ret is None:
            print('preprocess image failed!')
            exit(-1)
    else:
        print('No data_preprocess in yml')
        exit(-1)

    # inference with mlir framework
    input_npz = fp32_in_npz
    fp32_mlir_tensor_file = "{}_tensor_all_fp32.npz".format(model_name)
    output = net.inference('mlir', input_npz, mlirfile=fp32_mlirfile, model_file=model_file, weight_file=weight_file, all_tensors=fp32_mlir_tensor_file)
    if output is not None:
        print("mlir fp32 inference finish")

    # inference with origin framework
    fp32_origin_tensor_file = "{}_{}_tensor_all_fp32.npz".format(model_name, model_type)
    output = net.inference(model_type, input_npz, mlirfile=fp32_mlirfile, model_file=model_file, weight_file=weight_file, all_tensors=fp32_origin_tensor_file)
    if output is not None:
        print("{} fp32 inference finish".format(model_type))

    # accuracy fp32 test
    accuracy_test = config.get("Accuracy_test", None)
    if accuracy_test != None:
        excepts = accuracy_test.get("excepts", None)
        fp32_acc_test = accuracy_test.get("FP32_Accuracy_test", False)
        if fp32_acc_test:
            target_file = fp32_origin_tensor_file
            ref_file = fp32_mlir_tensor_file
            tolerance = accuracy_test.get('Tolerance_FP32')
            tolerance = "{},{},{}".format(tolerance[0], tolerance[1], tolerance[2])

            cvi_data_tool.npz_compare(target_file, ref_file,
                tolerance=tolerance,
                op_info=tpu_op_info,
                excepts=excepts
                )
            print("compare fp32 finish!")
    else:
        print("No FP32 interpreter accuracy test!")

    calibraion_table = "{}_threshold_table".format(model_name)
    # Calibration
    Calibration = config.get("Calibration", None)

    if Calibration:
        dataset_file = Calibration.get("Dataset")
        calibraion_table_in = Calibration.get("calibraion_table")

        if calibraion_table_in != None :  # use calibration_table directly.
            calibraion_table = calibraion_table_in
        else :  #do calibration
            image_num = Calibration.get("image_num", 1)
            histogram_bin_num = Calibration.get("histogram_bin_num", 2048)
            net.calibration(fp32_mlirfile, dataset_file, calibraion_table, preprocessor.run,image_num,histogram_bin_num)
    else:
        print("[Error]No Calibration in yml!")
        exit(-1)

    # build cvi_model
    cvimodel = config.get("output_file", None)
    Quantization = config.get("Quantization", None)
    if cvimodel:
        int8_mlirfile = "{}_int8.mlir".format(model_name)
        is_perchannel = Quantization.get("per_channel", True)
        is_symmetric = Quantization.get("symmetric", True)
        quant_tpu_op_info = "{}_quant_op_info.csv".format(model_name)

        net.build_cvimodel(fp32_mlirfile, cvimodel, calibraion_table, mlirfile_int8=int8_mlirfile, quant_info=quant_tpu_op_info)
    else:
        print("No cvimodel output_file")
        exit(-1)

    # Accuracy_test int8
    # inference with mlir framework
    int8_mlir_tensor_file = "{}_tensor_all_int8.npz".format(model_name)
    output = net.inference('mlir', input_npz, mlirfile=int8_mlirfile, model_file=None, weight_file=None, all_tensors=int8_mlir_tensor_file)
    if output is not None:
        print("mlir int8 inference finish")

    int8_acc_test = accuracy_test.get("INT8_Accuracy_test", False)
    if int8_acc_test:
        target_file = int8_mlir_tensor_file
        ref_file = fp32_origin_tensor_file #fp32_mlir_tensor_file
        tolerance = accuracy_test.get('Tolerance_INT8')
        tolerance = "{},{},{}".format(tolerance[0], tolerance[1], tolerance[2])

        cvi_data_tool.npz_compare(target_file, ref_file,
            tolerance=tolerance,
            op_info=quant_tpu_op_info,
            dequant=True,
            excepts=excepts,
            verbose=2
            )
        print("compare INT8 interpreter result finish!")
    else :
        print("No INT8 interpreter accuracy test!")

    Simulation = config.get("Simulation", None)
    if Simulation:
        simulation_tensor = "{}_tensor_all_simu.npz".format(model_name)

        # tpu_simulation
        net.tpu_simulation(input_npz, cvimodel, simulation_tensor, all_tensors=True)
        # compare with interprter
        cvi_data_tool.npz_compare(simulation_tensor, int8_mlir_tensor_file)
    else:
        print("No Simulation")

    # Clean
    net.cleanup()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_file",
                        help="YAML target file")
    args = parser.parse_args()
    with open(args.target_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
    parse(config)
    print("End to end finish")


if __name__ == "__main__":
    main()