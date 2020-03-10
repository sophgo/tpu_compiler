# system
import numpy as np
import logging
import argparse

from onnx import onnx, numpy_helper
from transform.onnx_converter import OnnxConverter
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-4s %(filename)2s %(lineno)d: %(message)s',
                    datefmt='%m-%d %H:%M')

def Test(args):
  # Create one input (ValueInfoProto)
  n, c, h, w = args.batch, args.channel, args.height, args.width
  factor = args.factor
  INPUT = np.arange(n * c * h * w).astype(np.float32)
  assert(c % (factor * factor) == 0)
  shape = [n, c / (factor*factor), factor, factor, h, w]

  # make input, half part set to < 0 for test relu case
  s = np.array_split(INPUT, 2)
  s[0] *= -1
  INPUT = np.concatenate(s)

  # reshape for real input
  INPUT = INPUT.reshape(n, c, h, w)
  np.savez(args.output_name, INPUT)
  return

  data = helper.make_tensor_value_info('data', TensorProto.FLOAT, shape)
  #value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, INPUT)
  #value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, list(INPUT.shape))

  # Create one output (ValueInfoProto)
  Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [n, c/(factor*factor), h * factor, w * factor])

  # Create a node (NodeProto) - This is based on Pad-11
  node_def = helper.make_node(
      args.node_name, # node name
      #['data', 'value'], # inputs
      ['data', 'value'], # inputs
      ['Y'], # outputs
      perm=[0, 1, 4, 2, 5, 3] # refer \onnx_converter.py
  )

  # Create the graph (GraphProto)
  graph_def = helper.make_graph(
      [node_def],
      'test-model',
      #[data, value],
      [data],
      [Y],
  )

  # Create the model (ModelProto)
  model_def = helper.make_model(graph_def, producer_name='onnx-test')

  c = OnnxConverter("test", model_def, "test.mlir")
  c.run()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_name",
      help="output fp32 file name, save with .npz",
      default="test_in_fp32.npz"
      )
  parser.add_argument(
      "--node_name",
      help="Node name for onnx, plz refer \OnnxConverte.py for more details"
      )
  parser.add_argument(
      "-n", "--batch",
      help="batch stand for 'n'",
      type=int
      )
  parser.add_argument(
      "-c", "--channel",
      help="channel stand for 'c'",
      type=int
      )
  parser.add_argument(
      "--height",
      help="height stand for 'h'",
      type=int
      )
  parser.add_argument(
      "-w", "--width",
      help="width stand for 'w'",
      type=int
      )
  parser.add_argument(
      "-f", "--factor",
      help="upsample factor for pixelshuufle need",
      type=int
      )

  args = parser.parse_args()
  Test(args)
