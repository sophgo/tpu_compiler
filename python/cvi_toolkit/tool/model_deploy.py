#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import time
import skimage
import caffe
import numpy as np
from cvi_toolkit.utils.log_setting import setup_logger
from cvi_toolkit.data.preprocess import preprocess
from cvi_toolkit.utils.mlir_shell import *
from cvi_toolkit.utils.intermediate_file import IntermediateFile
from cvi_toolkit.utils.mlir_parser import MlirParser

logger = setup_logger('root', log_level="INFO")

class DeployTool:
    def __init__(self, mlir_file, prefix):
        self.mlir_file = mlir_file
        self.prefix = prefix
        self.quantized_mlir = IntermediateFile(prefix, 'quantized.mlir')
        self.quantized_op_info_csv = IntermediateFile(prefix, 'quantized_op_info.csv', False)
        self.ppa = preprocess()
        self.ppa.load_config(self.mlir_file, 0)
        self.with_preprocess = False
        self.ppb = None
        self.mix = False

    def quantize(self, calib_table, mix_table, all_bf16, chip):
        if not calib_table and not mix_table and not all_bf16:
            self.mix = True
        ret = mlir_quant(self.mlir_file, calib_table, mix_table, all_bf16, chip,
                         str(self.quantized_mlir), str(self.quantized_op_info_csv))
        if ret != 0:
            raise RuntimeError("quantize fail")

    def fuse_preprocess(self, pixel_format, aligned_input):
        # prepare resize only input data
        config = {
          'resize_dims': ",".join([str(x) for x in self.ppa.resize_dims]),
          'keep_aspect_ratio': self.ppa.keep_aspect_ratio,
          'pixel_format': pixel_format,
          'aligned': aligned_input
        }
        self.ppb = preprocess()
        self.ppb.config(**config)

        fuse_preprocess_mlir = IntermediateFile(self.prefix, 'quantized_fuse_preprocess.mlir')
        ret = mlir_add_preprocess(str(self.quantized_mlir),
                                  str(fuse_preprocess_mlir),
                                  pixel_format, aligned_input)
        if ret != 0:
            raise RuntimeError("fuse preprocess fail")

        self.quantized_mlir = fuse_preprocess_mlir
        self.with_preprocess = True

    def build_cvimodel(self, cvimodel, dequant_results_to_fp32=True):
        IntermediateFile('_', 'lower_opt.mlir', False)
        IntermediateFile('_', 'final.mlir', False)
        ret = mlir_to_cvimodel(str(self.quantized_mlir), cvimodel, dequant_results_to_fp32)
        if ret != 0:
            raise RuntimeError("mlir to cvimodel failed")

    def get_batch_size(self, mlir_file):
        parser = MlirParser(mlir_file)
        return parser.get_batch_size(0)

    def _is_npz(self, image):
        return True if image.split('.')[-1] == 'npz' else False

    def validate(self, cvimodel, tolerance, excepts, correctness, image):
        batch_size = self.get_batch_size(str(self.quantized_mlir))
        # get all fp32 blobs of fp32 model by tpuc-interpreter
        in_fp32_npz = IntermediateFile(self.prefix, 'in_fp32.npz')
        if self._is_npz(image):
            x = np.load(image)
            np.savez(str(in_fp32_npz), **x)
        else:
            x = self.ppa.run(image, batch=batch_size)
            np.savez(str(in_fp32_npz), **{'input': x})

        blobs_interp_npz = IntermediateFile(self.prefix, 'full_precision_interp.npz', False)
        ret = mlir_inference(self.mlir_file, str(in_fp32_npz),
                             None, str(blobs_interp_npz))
        if ret != 0:
            raise RuntimeError("interpret fail")

        # get all quantized tensors of quantized model by tpuc-interpeter
        if self.with_preprocess:
            x = self.ppb.run(image, batch=batch_size)
            in_fp32_resize_only_npz = IntermediateFile(self.prefix, 'in_fp32_resize_only.npz')
            np.savez(str(in_fp32_resize_only_npz), **{'resize_only_data': x})
            in_fp32_npz = in_fp32_resize_only_npz
        all_tensors_interp_npz = IntermediateFile(self.prefix, 'quantized_tensors_interp.npz', False)
        ret = mlir_inference(str(self.quantized_mlir), str(in_fp32_npz), None,
                             str(all_tensors_interp_npz))
        if ret != 0:
            raise RuntimeError("interpret fail")

        # compare fp32 blobs and quantized tensors with tolerance similarity
        ret = fp32_blobs_compare(str(all_tensors_interp_npz),
                                 str(blobs_interp_npz),
                                 str(self.quantized_op_info_csv),
                                 tolerance, dequant=True,
                                 excepts=excepts)
        if ret != 0:
            raise RuntimeError("validate fail")

        if not cvimodel:
            return

        # compare quantized tensors, which generated from simulator and
        # tpuc-interpreter
        all_tensors_sim_npz = IntermediateFile(self.prefix, 'quantized_tensors_sim.npz', True)
        ret = run_cvimodel(str(in_fp32_npz), cvimodel,
                           str(all_tensors_sim_npz),
                           all_tensors=True)
        if ret != 0:
            raise RuntimeError("run simulator fail")

        ret = fp32_blobs_compare(str(all_tensors_sim_npz),
                                 str(all_tensors_interp_npz),
                                 str(self.quantized_op_info_csv),
                                 tolerance=correctness,
                                 show_detail=True,
                                 int8_tensor_close=self.mix)
        if ret != 0:
            raise RuntimeError("validate fail")

    def cleanup(self):
        IntermediateFile.cleanup()


if __name__ == '__main__':
    # fix bool bug of argparse
    def str2bool(v):
      return v.lower() in ("yes", "true", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model_name")
    parser.add_argument("--mlir", required=True, help="optimized mlir fp32 model")
    parser.add_argument("--calibration_table", help="calibration table for int8 quantization")
    parser.add_argument("--mix_precision_table", help="table of OPs that quantized to bf16")
    parser.add_argument("--all_bf16", action='store_true', help="quantize all OPs to bf16")
    parser.add_argument("--tolerance", required=True, help="tolerance")
    parser.add_argument("--excepts", default='-', help="excepts")
    parser.add_argument("--correctness", default='0.99,0.99,0.98', help="correctness")
    parser.add_argument("--chip", required=True, choices=['cv183x', 'cv182x'], help="chip platform name")
    parser.add_argument("--fuse_preprocess", action='store_true', default=False,
                        help="add tpu preprocesses (mean/scale/channel_swap) in the front of model")
    parser.add_argument("--pixel_format", help="pixel format of input frame to the model")
    parser.add_argument("--aligned_input", type=str2bool, default=False,
                        help='if the input frame is width/channel aligned')
    parser.add_argument("--dequant_results_to_fp32", type=str2bool, default=True,
                        help="if dequant all results to fp32")
    parser.add_argument("--image", required=True, help="input image or npz file for inference")
    parser.add_argument("--cvimodel", help='output cvimodel')
    parser.add_argument("--debug", action='store_true', help='to keep all intermediate files for debug')
    args = parser.parse_args()

    if args.cvimodel:
        prefix = args.cvimodel.split("/")[-1]
        prefix = prefix.replace('.cvimodel', '')
    else:
        prefix = args.model_name
    tool = DeployTool(args.mlir, prefix)
    tool.quantize(args.calibration_table,
                  args.mix_precision_table,
                  args.all_bf16,
                  args.chip)
    if args.fuse_preprocess:
        tool.fuse_preprocess(args.pixel_format,
                             args.aligned_input)
    if args.cvimodel:
        tool.build_cvimodel(args.cvimodel, args.dequant_results_to_fp32)
    tool.validate(args.cvimodel, args.tolerance, args.excepts,
                  args.correctness, args.image)
    if not args.debug:
        tool.cleanup()
