#!/usr/bin/python3
import yaml
import sys, re, os
import argparse
from cvi_toolkit import cvinn, preprocess
from cvi_toolkit.numpy_helper import npz_extract, npz_rename
from cvi_toolkit import cvi_data
net = cvinn()
cvi_data_tool = cvi_data()
preprocessor = preprocess()

# close caffe glog
os.environ['GLOG_minloglevel'] = "2"
os.environ['GLOG_log_dir'] = "."
os.environ['GLOG_logbufsecs'] = "1"
os.environ['GLOG_logtostderr']="0"
os.environ['GLOG_stderrthreshold'] = "3"

env_matcher = re.compile(r'\$\{([^}^{]+)\}')
def env_constructor(loader, node):
  ''' Extract the matched value, expand env variable, and replace the match '''
  value = node.value
  match = env_matcher.match(value)
  env_var = match.group()[2:-1]
  return os.environ.get(env_var) + value[match.end():]

yaml.add_implicit_resolver('!path', env_matcher,  None, yaml.SafeLoader)
yaml.add_constructor('!path', env_constructor, yaml.SafeLoader)


def parse(config: dict):
    # model to mlir
    model_name = None
    Convert_model = config.get('Convert_model', None)
    if Convert_model:
        t = Convert_model
        model_type = t.get('framework_type')
        model_file = t.get('model_file')
        weight_file = t.get('weight_file', None)

        model_name = model_file.split('.')[0].split('/')[-1]
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

        if mean != None and channel_mean != None:
            print('[Warning] mean and channel both exist, use channel mean')
            mean = channel_mean

        mean_file = t.get('mean_file', None)
        if mean_file != None and mean != None:
            print('[Warning] mean and mean_file both exist, use mean_file')
            mean=None

        input_scale = t.get('input_scale', 1.0)
        net_input_dims = t.get('net_input_dims', "224,224")
        raw_scale = t.get('raw_scale', 255.0)
        channel_swap = t.get('channel_swap', '2,1,0')
        resize_dims = t.get('image_resize_dim', "256,256")
        letter_box = t.get('LetterBox', None)


        input_file = t['input_file']
        output_npz = t['output_npz']
        fp32_in_npz = output_npz
        preprocessor.config(net_input_dims=net_input_dims,
                    resize_dims=resize_dims,
                    mean=mean,
                    mean_file=mean_file,
                    input_scale=input_scale,
                    raw_scale=raw_scale,
                    channel_swap=channel_swap,
                    letter_box=None)

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
        fp32_acc_test = accuracy_test.get("FP32_Accuracy_test", False)
        if fp32_acc_test:
            target_file = fp32_origin_tensor_file
            ref_file = fp32_mlir_tensor_file
            tolerance = accuracy_test.get('Tolerance_FP32')
            tolerance = "{},{},{}".format(tolerance[0], tolerance[1], tolerance[2])

            cvi_data_tool.npz_compare(target_file, ref_file,
                tolerance=tolerance,
                op_info=tpu_op_info
                )
            print("compare fp32 finish!")
    else:
        print("No acc test")

    calibraion_table = "{}_threshold_table".format(model_name)
    # Calibration
    Calibration = config.get("Calibration", None)
    if Calibration:
        dataset_file = Calibration.get("Dataset")
        net.calibration(fp32_mlirfile, dataset_file, calibraion_table, preprocessor.run)
    else:
        print("No Calibration at yaml")


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

        elif cmd == "cvi_npz_rename":
            input_npz = t['input_npz']
            target_name = t['target_name']
            ref_name = t['ref_name']
            npz_rename([input_npz, target_name, ref_name])

        elif cmd == "cvi_npz_compare":

            target_file = t['target_file']
            ref_file = t['ref_file']
            verbose = t.get('verbose', 0)
            discard = t.get('discard', 0)
            dtype = t.get('dtype', "")
            tolerance= t.get('tolerance', "0.99,0.99,0.90")
            op_info = t.get('op_info', None)
            order = t.get('order', None)
            tensor = t.get('tensor', None)
            excepts = t.get('excepts', None)
            save = t.get('save', None)
            dequant = t.get('dequant', False)
            full_array = t.get('full_array', False)
            stats_int8_tensor = t.get('stats_int8_tensor', False)

            cvi_data_tool.npz_compare(target_file, ref_file,
                verbose=verbose,
                discard=discard,
                dtype=dtype,
                tolerance=tolerance,
                op_info=op_info,
                order=order,
                tensor=tensor,
                excepts=excepts,
                save=save,
                dequant=dequant,
                full_array=full_array,
                stats_int8_tensor=stats_int8_tensor
                )



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
