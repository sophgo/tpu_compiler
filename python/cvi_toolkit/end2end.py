import yaml
import sys, re, os
import argparse
from cvi_toolkit import cvinn, preprocess
from cvi_toolkit.numpy_helper import npz_extract, npz_rename
from cvi_toolkit import cvi_data
net = cvinn()
cvi_data_tool = cvi_data()
preprocessor = preprocess()

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
    task = config.get('cvi_config', None)
    for t in task:
        cmd = t.get('cmd', None)
        if cmd == 'load_model':
            model_type = t['model_type']
            model_file = t['model_file']
            weight_file = t.get('weight_file', None)
            mlirfile = t['mlirfile']
            tpu_op_info = t.get('tpu_op_info', "")

            ret = net.load_model(model_type, model_file, mlirfile, weight_file=weight_file, tpu_op_info=tpu_op_info)
            if ret != 0:
                print('load model failed!')
                exit(-1)

        elif cmd == "preprocess":
            mean = t['mean']
            mean_file = t.get('mean_file', None)
            input_scale = t.get('input_scale', 1.0)
            net_input_dims = t.get('net_input_dims', None)
            raw_scale = t.get('raw_scale', 255.0)
            channel_swap = t.get('channel_swap', '2,1,0')
            resize_dims = t.get('resize_dims', "256,256")
            letter_box = t.get('letter_box', None)


            input_file = t['input_file']
            output_npz = t['output_npz']

            preprocessor.config(net_input_dims=net_input_dims,
                     resize_dims=resize_dims,
                     mean=mean,
                     mean_file=None,
                     input_scale=input_scale,
                     raw_scale=raw_scale,
                     channel_swap=channel_swap,
                     letter_box=None)

            preprocessor.run(input_file, output_npz)

        elif cmd == "inference":
            model_type = t['model_type']
            model_file = t.get('model_file', None)
            weight_file = t.get('weight_file', None)
            mlirfile = t.get('mlirfile', None)
            input_npz = t['input_npz']
            all_tensors = t.get('all_tensors', None)

            net.inference(model_type, input_npz, mlirfile=mlirfile, model_file=model_file, weight_file=weight_file, all_tensors=all_tensors)

        elif cmd == "build_cvimodel":
            mlirfile_fp32 = t['mlirfile_fp32']
            cvimodel = t['cvimodel']
            threshold_table = t['threshold_table']
            mlirfile_int8 = t.get('mlirfile_int8', None)
            quant_method = t.get('perchannel', 'perchannel')
            cmd_buf = t.get('cmd_buf', 'cmdbuf.bin')
            quant_info = t.get('quant_info', None)

            net.build_cvimodel(mlirfile_fp32, cvimodel, threshold_table, mlirfile_int8=mlirfile_int8,
                        quant_method=quant_method, cmd_buf=cmd_buf, quant_info=quant_info)

        elif cmd == "tpu_simulation":
            input_file = t['input_file']
            cvimodel = t['cvimodel']
            output_tensor = t['output_tensor']
            all_tensors = t.get('all_tensors', True)

            net.tpu_simulation(input_file, cvimodel, output_tensor, all_tensors=all_tensors)

        elif cmd == "cvi_npz_extract":
            input_npz = t['input_npz']
            ouput_npz = t['ouput_npz']
            extract_name = t['extract_name']
            npz_extract([input_npz, ouput_npz, extract_name])

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
