import argparse
import os, cv2
from model.retinaface.retinaface_util import RetinaFace
from kld_calibrator import KLD_Calibrator
from asym_calibrator import Asym_Calibrator

g_net_input_dims = [600, 600]


def preprocess_func(image_path):
  retinaface_w, retinaface_h = g_net_input_dims[0], g_net_input_dims[1]
  detector = RetinaFace()
  image = cv2.imread(str(image_path).rstrip())
  x = detector.preprocess(image, retinaface_w, retinaface_h)
  return x


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_name', metavar='model-name', help='Model name')
  parser.add_argument('model_file', metavar='model-path', help='Model path')
  parser.add_argument('image_list_file', metavar='image_list_file', help='Input image list file')
  parser.add_argument('output_file', metavar='output_file', help='Output threshold table file')
  parser.add_argument('--calibrator', metavar='calibrator', help='Calibration method', default='KLD')
  parser.add_argument('--out_path', metavar='path', help='Output path', default='./result')
  parser.add_argument('--math_lib_path', metavar='math_path', help='Calibration math library path', default='calibration_math.so')
  parser.add_argument('--input_num', metavar='input_num', help='Calibration data number', default=10)
  parser.add_argument('--histogram_bin_num', metavar='histogram_bin_num', help='Specify histogram bin numer for kld calculate',
                      default=2048)
  parser.add_argument('--auto_tune', action='store_true', help='Enable auto tune or not')
  parser.add_argument('--binary_path', metavar='binary_path', help='MLIR binary path')
  parser.add_argument('--tune_iteration', metavar='iteration', help='The number of data using in tuning process', default=10)
  parser.add_argument("--net_input_dims", default='600,600',
                      help="'height,width' dimensions of net input tensors.")
  args = parser.parse_args()

  g_net_input_dims = [int(s) for s in args.net_input_dims.split(',')]

  if not os.path.isdir(args.out_path):
    os.mkdir(args.out_path)

  if args.calibrator == 'KLD':
    calibrator = KLD_Calibrator(args, preprocess_func)
  elif args.calibrator == 'Asym':
    calibrator = Asym_Calibrator(args, preprocess_func)
  thresholds = calibrator.do_calibration()
  calibrator.dump_threshold_table(args.output_file, thresholds)

