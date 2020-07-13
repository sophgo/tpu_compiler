#!/usr/bin/env python3
import sys, re, os
import yaml
import argparse
import site

import shutil



from cvi_toolkit import cvinn, preprocess
from cvi_toolkit.numpy_helper import npz_extract, npz_rename
from cvi_toolkit import cvi_data
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root', log_level="INFO")

net = cvinn()
cvi_data_tool = cvi_data()
preprocessor = preprocess()

def check_file_exist(filename):
    if not os.path.exists(filename):
        logger.error("File \"{}\" not existed!".format(filename))
        return False
    return True

def check_file_assert(filename):
    if filename != None and not check_file_exist(filename):
        exit(-1)

def parse(config: dict):
    # model to mlir
    model_name = None
    output_file = config.get("output_file", None)
    if os.path.exists(output_file):
        logger.info("remove already existed {}".format(output_file))
        os.remove(output_file)
    os.makedirs("tmp", exist_ok=True)
    os.chdir("tmp")

    model_name = output_file.split('.')[0].split('/')[-1]
    Convert_model = config.get('Convert_model', None)
    chip = config.get('chip', 'cv183x')
    if Convert_model:
        t = Convert_model
        model_type = t.get('framework_type')
        model_file = t.get('model_file')
        weight_file = t.get('weight_file', None)
        tpu_op_info = "{}_op_info.csv".format(model_name)

        check_file_assert(model_file)
        check_file_assert(weight_file)

        fp32_mlirfile = "{}.mlir".format(model_name)
        try:
            logger.info("convert model to fp32 mlir ...")
            ret = net.convert_model(model_type, model_file, fp32_mlirfile, weight_file=weight_file, tpu_op_info=tpu_op_info,batch_size=1, chip=chip)
            if ret != 0:
                logger.error("mlir_translate failed")
                exit(-1)
        except RuntimeError as err:
            logger.error("RuntimeError {}".format(err))
            exit(-1)
        logger.info("convert model to fp32 mlir finished")
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
        check_file_assert(mean_file)

        if mean != None and channel_mean != None:
            logger.error("channel_mean and mean should not be set at the same time!")
            exit(-1)
        elif channel_mean != None:
            mean = channel_mean

        if mean != None and mean_file != None:
            logger.error("mean_file and mean value should not be set at the same time!")
            exit(-1)

        input_scale = t.get('input_scale', 1.0)
        net_input_dims = t.get('net_input_dims')
        if net_input_dims == None :
            logger.error("net_input_dims should be set in yml file !!")
            exit(-1)
        std = t.get('std', "1,1,1")
        raw_scale = t.get('raw_scale', 255.0)
        data_format = t.get('data_format')
        resize_dims = t.get('image_resize_dim')
        letter_box = t.get('LetterBox', False)

        rgb_order = t.get('RGB_order',"bgr")
        npy_input = t.get('npy_input', None)

        check_file_assert(npy_input)

        input_file = t.get('input_file', None)
        if not npy_input:
            # if has npy_input, we don't need input_file
            check_file_assert(input_file)
        if input_file == None and npy_input == None:
            logger.error('Please set input file image in yml!')
            exit(-1)

        output_npz = t['output_npz']
        fp32_in_npz = output_npz
        preprocessor.config(net_input_dims=net_input_dims,
                    resize_dims=resize_dims,
                    mean=mean,
                    mean_file=mean_file,
                    input_scale=input_scale,
                    std=std,
                    raw_scale=raw_scale,
                    data_format=data_format,
                    rgb_order=rgb_order,
                    npy_input=npy_input,
                    letter_box=letter_box)

        ret = preprocessor.run(input_file, output_npz=output_npz,
                               output_channel_order=rgb_order)

        if ret is None:
            logger.error('preprocess image failed!')
            exit(-1)
    else:
        logger.error('No data_preprocess in yml')
        exit(-1)

    input_npz = fp32_in_npz
    # inference with origin framework
    fp32_origin_tensor_file = "{}_origin_tensor_all_fp32.npz".format(
        model_name)
    logger.info("run original {} model fp32 inference ...".format(model_type))
    output = net.inference(model_type, input_npz, mlirfile=fp32_mlirfile,
                           model_file=model_file, weight_file=weight_file, all_tensors=fp32_origin_tensor_file)
    if output is not None:
        logger.info(
            "original {} model fp32 inference finished".format(model_type))

    # inference with mlir framework
    logger.info("run mlir fp32 inference ...")
    fp32_mlir_tensor_file = "{}_tensor_all_mlir_fp32.npz".format(model_name)
    try:
        if data_format == "nhwc":
            # beacause mlir data_format is nchw,
            # transpose it
            cvi_data_tool.npz_transpose(input_npz, "nhwc", "nchw")
            cvi_data_tool.npz_transpose(
                fp32_origin_tensor_file, "nhwc", "nchw")
        output = net.inference('mlir', input_npz, mlirfile=fp32_mlirfile, model_file=model_file, weight_file=weight_file, all_tensors=fp32_mlir_tensor_file)
    except Exception as e:
        logger.error("mlir fp32 inference failed {}".format(e))
        exit(-1)
    if output is not None:
        logger.info("mlir fp32 inference finished")


    # accuracy fp32 test
    accuracy_test = config.get("Accuracy_test", None)
    if accuracy_test != None:
        excepts = accuracy_test.get("excepts", None)
        logger.info("run mlir fp32 accuracy test ...")
        fp32_acc_test = accuracy_test.get("FP32_Accuracy_test", False)
        if fp32_acc_test:
            target_file = fp32_origin_tensor_file
            ref_file = fp32_mlir_tensor_file
            tolerance = accuracy_test.get('Tolerance_FP32')
            tolerance = "{},{},{}".format(tolerance[0], tolerance[1], tolerance[2])

            fp32_stat = cvi_data_tool.npz_compare(target_file, ref_file,
                tolerance=tolerance,
                op_info=tpu_op_info,
                excepts=excepts,
                verbose=2
                )
            logger.info("mlir fp32 accuracy test finished")
    else:
        logger.warning("No FP32 interpreter accuracy test!")

    calibraion_table = "{}_threshold_table".format(model_name)
    # Calibration
    Calibration = config.get("Calibration", None)
    new_table = False
    if Calibration:
        dataset_file = Calibration.get("Dataset")
        calibraion_table_in = Calibration.get("calibraion_table", None)
        check_file_assert(calibraion_table_in)

        auto_tune = Calibration.get("auto_tune", False)
        if calibraion_table_in != None :  # use calibration_table directly.
            logger.info("import calibration table")
            calibraion_table = calibraion_table_in
        else :
            new_table = True
            check_file_assert(dataset_file)
            if os.path.exists(calibraion_table):
                logger.info("remove already existed {}".format(calibraion_table))
                os.remove(calibraion_table)
            # if no callibration table do calibration
            logger.info("run calibration ...")
            image_num = Calibration.get("image_num", 1)
            tune_image_num = Calibration.get("tune_image_num", 10)

            histogram_bin_num = Calibration.get("histogram_bin_num", 2048)
            if data_format == "nhwc":
                # mlir data_format is nchw, change it
                preprocessor.data_format = "nchw"
            net.calibration(fp32_mlirfile, dataset_file, calibraion_table, preprocessor.run, image_num, histogram_bin_num, auto_tune=auto_tune,tune_image_num=tune_image_num)
            logger.info("calibration finished")
    else:
        logger.error("No Calibration in yml!")
        exit(-1)

    # build cvi_model
    cvimodel = config.get("output_file", None)
    Quantization = config.get("Quantization", None)
    if cvimodel:
        logger.info("run cvimodel generation ...")
        int8_mlirfile = "{}_int8.mlir".format(model_name)
        is_perchannel = Quantization.get("per_channel", True)
        is_symmetric = Quantization.get("symmetric", True)
        quant_tpu_op_info = "{}_quant_op_info.csv".format(model_name)

        net.build_cvimodel(fp32_mlirfile, cvimodel, calibraion_table, mlirfile_int8=int8_mlirfile, quant_info=quant_tpu_op_info)
        logger.info("cvimodel generation finished")
    else:
        logger.error("No cvimodel output_file")
        exit(-1)

    # Accuracy_test int8
    # inference with mlir framework
    int8_mlir_tensor_file = "{}_tensor_all_int8.npz".format(model_name)
    logger.info("run mlir int8 inference ...")
    output = net.inference('mlir', input_npz, mlirfile=int8_mlirfile, model_file=None, weight_file=None, all_tensors=int8_mlir_tensor_file)
    if output is not None:
        logger.info("mlir int8 inference finished")

    int8_acc_test = accuracy_test.get("INT8_Accuracy_test", False)
    if int8_acc_test:
        target_file = int8_mlir_tensor_file
        ref_file = fp32_origin_tensor_file #fp32_mlir_tensor_file
        tolerance = accuracy_test.get('Tolerance_INT8')
        tolerance = "{},{},{}".format(tolerance[0], tolerance[1], tolerance[2])
        logger.info("run int8 interpreter accurracy test ...")
        int8_stat = cvi_data_tool.npz_compare(target_file, ref_file,
            tolerance=tolerance,
            op_info=quant_tpu_op_info,
            dequant=True,
            excepts=excepts,
            verbose=2
            )
        logger.info("int8 interpreter accurracy test finished")
    else :
        logger.error("No INT8 interpreter accuracy test!")

    Simulation = config.get("Simulation", None)
    if Simulation:
        simulation_tensor = "{}_tensor_all_simu.npz".format(model_name)
        logger.info("run cvimodel TPU HW inference simulation ...")
        # tpu_simulation
        ret = net.tpu_simulation(input_npz, cvimodel, simulation_tensor, all_tensors=True)
        if ret != 0:
            logger.error("Simulation cvimodel Failed!")
            exit(-1)
        # compare with interprter
        cvi_data_tool.npz_compare(simulation_tensor, int8_mlir_tensor_file)
        logger.info("cvimodel TPU HW inference simulation finished")
    else:
        logger.warning("No cvimodel TPU HW simulation")

    # Clean
    logger.info("run cleanup ...")
    net.cleanup()
    shutil.move(output_file, '../')
    if new_table:
        if os.path.exists('../{}'.format(calibraion_table)):
            logger.info("remove already existed {}".format(calibraion_table))
            os.remove('../{}'.format(calibraion_table))
        shutil.move(calibraion_table, '../')
    os.chdir("../")
    shutil.rmtree("tmp", ignore_errors=True)
    logger.info("cleanup finished")
    if fp32_acc_test:
        logger.info("float32 tensor min_similiarity = ({}, {}, {})".format(
            fp32_stat.min_cosine_similarity,
            fp32_stat.min_correlation_similarity,
            fp32_stat.min_euclidean_similarity))
    if int8_acc_test:
        logger.info("int8 tensor min_similiarity = ({}, {}, {})".format(
            int8_stat.min_cosine_similarity,
            int8_stat.min_correlation_similarity,
            int8_stat.min_euclidean_similarity))

    logger.info("You can get cvimodel:\n {}".format(os.path.abspath(output_file)))
    if new_table:
        logger.info("New Threshold Table:\n {}".format(os.path.abspath(calibraion_table)))


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


if __name__ == "__main__":
    main()
