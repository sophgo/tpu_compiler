#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import time
import skimage
import caffe
import numpy as np
from cvi_toolkit.utils.version import declare_toolchain_version
from cvi_toolkit.utils.log_setting import setup_logger
from cvi_toolkit.data.preprocess import preprocess, supported_pixel_formats
from cvi_toolkit.utils.mlir_shell import *
from cvi_toolkit.utils.intermediate_file import IntermediateFile
from cvi_toolkit.utils.mlir_parser import MlirParser

logger = setup_logger('root', log_level="INFO")


def check_return_value(cond, msg):
    if not cond:
        raise RuntimeError(msg)


class DeployTool:
    def __init__(self, mlir_file, prefix):
        self.mlir_file = mlir_file
        self.prefix = prefix
        self.quantized_mlir = IntermediateFile(prefix, 'quantized.mlir')
        self.quantized_op_info_csv = IntermediateFile(prefix, 'quantized_op_info.csv', False)
        self.in_fp32_npz = IntermediateFile(prefix, 'in_fp32.npz')
        self.in_fp32_resize_only_npz = IntermediateFile(prefix, 'in_fp32_resize_only.npz')
        self.all_tensors_interp_npz = IntermediateFile(prefix, 'quantized_tensors_interp.npz', False)
        self.ppa = preprocess()
        self.input_num = self.ppa.get_input_num(self.mlir_file)
        self.with_preprocess = False
        self.pixel_format = 'BGR_PLANAR'
        self.aligned_input = False

    def fuse_preprocess(self, pixel_format, aligned_input):
        fuse_preprocess_mlir = IntermediateFile(self.prefix, 'quantized_fuse_preprocess.mlir')
        ret = mlir_add_preprocess(str(self.quantized_mlir),
                                  str(fuse_preprocess_mlir),
                                  pixel_format, aligned_input)
        check_return_value(ret == 0, "fuse preprocess failed")

        self.quantized_mlir = fuse_preprocess_mlir
        self.with_preprocess = True
        self.pixel_format = pixel_format
        self.aligned_input = aligned_input

    def quantize(self, calib_table, mix_table, all_bf16, chip,
                 fuse_preprocess, pixel_format, aligned_input, quantize=""):
        self.chip = chip
        ret = mlir_quant(self.mlir_file, str(self.quantized_mlir),
                         chip, str(self.quantized_op_info_csv),
                         all_bf16, calib_table, mix_table, quantize)
        check_return_value(ret == 0, 'quantization failed')

        if fuse_preprocess:
            self.fuse_preprocess(pixel_format, aligned_input)

    def _get_batch_size(self, mlir_file):
        parser = MlirParser(mlir_file)
        return parser.get_batch_size(0)

    @staticmethod
    def _is_npz(image):
        return True if image.split('.')[-1] == 'npz' else False

    @staticmethod
    def _is_npy(image):
        return True if image.split('.')[-1] == 'npy' else False

    def _prepare_input_npz(self, images):
        batch_size = self._get_batch_size(str(self.quantized_mlir))
        # get all fp32 blobs of fp32 model by tpuc-interpreter
        if len(images) == 1 and self._is_npz(images[0]):
            x = np.load(images[0])
            np.savez(str(self.in_fp32_npz), **x)
            if self.with_preprocess:
                np.savez(str(self.in_fp32_resize_only_npz), **x)
        else:
            x0 = {}
            x1 = {}
            assert(len(images) == self.input_num)
            for i in range(self.input_num):
                self.ppa.load_config(self.mlir_file, i, self.chip)
                if self._is_npy(images[i]):
                    data = np.load(images[i])
                    x0[self.ppa.input_name] = data
                    x1[self.ppa.input_name] = data
                    continue
                x0[self.ppa.input_name] = self.ppa.run(images[i], batch=batch_size)
                if self.with_preprocess:
                    config = {
                        'resize_dims': ",".join([str(x) for x in self.ppa.resize_dims]),
                        'keep_aspect_ratio': self.ppa.keep_aspect_ratio,
                        'pixel_format': self.pixel_format,
                        'aligned': self.aligned_input,
                        'chip': self.chip,
                    }
                    ppb = preprocess()
                    ppb.config(**config)
                    x1[self.ppa.input_name] = ppb.run(images[i], batch=batch_size)

            np.savez(str(self.in_fp32_npz), **x0)
            if self.with_preprocess:
                np.savez(str(self.in_fp32_resize_only_npz), **x1)

    def validate_quantized_model(self, tolerance, excepts, images, custom_op_plugin):
        self._prepare_input_npz(images)
        blobs_interp_npz = IntermediateFile(self.prefix, 'full_precision_interp.npz', False)
        ret = mlir_inference(self.mlir_file, str(self.in_fp32_npz),
                             None, str(blobs_interp_npz), custom_op_plugin)
        check_return_value(ret == 0, "inference of fp32 model failed")

        # get all quantized tensors of quantized model by tpuc-interpeter
        in_fp32_npz = self.in_fp32_npz
        if self.with_preprocess:
            in_fp32_npz = self.in_fp32_resize_only_npz

        ret = mlir_inference(str(self.quantized_mlir), str(in_fp32_npz), None,
                             str(self.all_tensors_interp_npz), custom_op_plugin)
        check_return_value(ret == 0, "inference of quantized model failed")

        # compare fp32 blobs and quantized tensors with tolerance similarity
        ret = fp32_blobs_compare(str(self.all_tensors_interp_npz),
                                 str(blobs_interp_npz),
                                 str(self.quantized_op_info_csv),
                                 tolerance, dequant=True,
                                 excepts=excepts)
        check_return_value(ret == 0, "accuracy validation of quantized model failed")

    def build_cvimodel(self, cvimodel, inputs_type="AUTO", outputs_type="FP32",
                       append_weight=False, tg_op_divide=False,
                       model_version="", custom_op_plugin=""):
        IntermediateFile('_', 'lower_opt.mlir', False)
        IntermediateFile('_', 'final.mlir', True)
        if model_version == "":
            model_version = "latest"
        ret = mlir_to_cvimodel(str(self.quantized_mlir), cvimodel, inputs_type,
                               outputs_type, append_weight, tg_op_divide, model_version, custom_op_plugin)
        check_return_value(ret == 0, "failed to generate cvimodel")

    def validate_cvimodel(self, cvimodel, correctness, excepts):
        in_fp32_npz = self.in_fp32_npz
        if self.with_preprocess:
            in_fp32_npz = self.in_fp32_resize_only_npz
        # compare quantized tensors, which generated from simulator and
        # tpuc-interpreter
        all_tensors_sim_npz = IntermediateFile(self.prefix, 'quantized_tensors_sim.npz', True)
        ret = run_cvimodel(str(in_fp32_npz), cvimodel,
                           str(all_tensors_sim_npz),
                           all_tensors=True)
        check_return_value(ret == 0, "run cvimodel in simulator failed")

        ret = fp32_blobs_compare(str(all_tensors_sim_npz),
                                 str(self.all_tensors_interp_npz),
                                 str(self.quantized_op_info_csv),
                                 tolerance=correctness,
                                 excepts=excepts,
                                 show_detail=True,
                                 mix_precision=True)
        check_return_value(ret == 0, "accuracy validation of cvimodel failed")

    def cleanup(self):
        IntermediateFile.cleanup()


# fix bool bug of argparse
def str2bool(v):
    return v.lower() in ("yes", "true", "1")

def deprecated_option(cond, msg):
    if cond:
        raise RuntimeError(msg)

if __name__ == '__main__':
    declare_toolchain_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model_name")
    parser.add_argument("--mlir", required=True, help="optimized mlir fp32 model")
    parser.add_argument("--calibration_table", help="calibration table for int8 quantization")
    parser.add_argument("--mix_precision_table", help="table of OPs that quantized to specific mode")
    parser.add_argument("--quantize", default='', help="set qauntization type: BF16/INT8/MIX_BF16")
    parser.add_argument("--tolerance", required=True, help="tolerance")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--correctness", default='0.99,0.99,0.98', help="correctness")
    parser.add_argument("--chip", required=True, choices=['cv183x', 'cv182x', 'mars'], help="chip platform name")
    parser.add_argument("--fuse_preprocess", action='store_true', default=False,
                        help="add tpu preprocesses (mean/scale/channel_swap) in the front of model")
    parser.add_argument("--pixel_format", choices=supported_pixel_formats, default='BGR_PLANAR',
                        help="pixel format of input frame to the model")
    parser.add_argument("--aligned_input", type=str2bool, default=False,
                        help='if the input frame is width/channel aligned')
    parser.add_argument("--inputs_type", default="AUTO",
                        help="set inputs type:AUTO/FP32/INT8/BF16/SAME; if AUTO, use INT8 if input layer is INT8, use FP32 if BF16")
    parser.add_argument("--outputs_type", default="FP32",
                        help="set outputs type:AUTO/FP32/INT8/BF16/SAME; if AUTO, use INT8 if output layer is INT8, use FP32 if BF16")
    parser.add_argument("--merge_weight", action='store_true',
                        help="merge weights into one weight binary wight previous generated cvimodel")
    parser.add_argument("--tg_op_divide", type=str2bool, default=False,
                        help="if divide tg ops to save gmem")
    parser.add_argument("--model_version", default="latest",
                        help="if need old version cvimodel, set the verion, such as 1.2")
    parser.add_argument("--custom_op_plugin", default="",
                        help="custom op plugin so file path")
    parser.add_argument("--image", default=None, help="input image/npz/npy file for inference, "
                       "if has more than one input images, join images with semicolon")
    parser.add_argument("--cvimodel", required=True, help='output cvimodel')
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    #### DEPRECATED
    parser.add_argument("--all_bf16", action='store_true', help="DEPRECATED, please use --quantize BF16")
    parser.add_argument("--dequant_results_to_fp32", action='store_true', help="DEPRECATED, please use --outputs_type FP32")
    parser.add_argument("--expose_bf16_inputs", action='store_true', help="DEPRECATED, please use --inputs_type BF16")
    parser.add_argument("--compress_weight", action='store_true', help="DEPRECATED, no need any more")
    args = parser.parse_args()
    ##check options DEPRECATED
    deprecated_option(args.dequant_results_to_fp32, "DEPRECATED, please use --outputs_type FP32")
    deprecated_option(args.expose_bf16_inputs, "DEPRECATED, please use --inputs_type BF16")
    deprecated_option(args.compress_weight, "DEPRECATED, no need any more")
    #deprecated_option(args.all_bf16, "DEPRECATED, please use --quantize BF16")

    prefix = args.model_name
    tool = DeployTool(args.mlir, prefix)
    # quantize and validate accuracy
    tool.quantize(args.calibration_table,
                  args.mix_precision_table,
                  args.all_bf16,
                  args.chip,
                  args.fuse_preprocess,
                  args.pixel_format,
                  args.aligned_input,
                  args.quantize)
    if args.image:
        images = args.image.split(',')
        images = [s.strip() for s in images]
        tool.validate_quantized_model(args.tolerance, args.excepts, images, args.custom_op_plugin)

    # generate cvimodel and validate accuracy
    tool.build_cvimodel(args.cvimodel, args.inputs_type, args.outputs_type,
                        args.merge_weight, args.tg_op_divide, args.model_version, args.custom_op_plugin)
    if args.image:
        tool.validate_cvimodel(args.cvimodel, args.correctness, args.excepts)

    if not args.debug:
        tool.cleanup()
