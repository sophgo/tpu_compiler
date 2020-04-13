#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import skimage
import caffe


class My_Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file,
                 image_dims=None, mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None, batch_size=1):
        caffe.Net.__init__(self, model_file, caffe.TEST, weights=pretrained_file)

        # configure pre-processing
        in_ = self.inputs[0]

        self.blobs[in_].reshape(batch_size, self.blobs[in_].data.shape[1], self.blobs[in_].data.shape[2], self.blobs[in_].data.shape[3])
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

        # dump net layers
        print("Network layers:")
        for name, layer in zip(self._layer_names, self.layers):
            print("{:<27}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
        print("Blobs:")
        for name, blob in self.blobs.items():
            print("{:<27}:  {}".format(name, blob.data.shape))
        print("Weights:")
        for name, param in self.params.items():
            # print("{:<27}:  {}".format(name, param[0].data.shape))
            for p in param:
              print("{:<27}:  {}".format(name, p.data.shape))

    def predict(self, inputs, dump_blobs=None, dump_weights=None,
                force_input=None):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.

        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        # Take center crop.
        center = np.array(self.image_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([
            -self.crop_dims / 2.0,
            self.crop_dims / 2.0
        ])
        crop = crop.astype(int)
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)

        # [DEBUG] turn on to replace the input data
        if force_input is not None:
            print("replace input data with ", force_input)
            caffe_in[0] = np.load(force_input)

        # [DEBUG] dump input to file
        # print("caffe_in[0].shape", caffe_in[0].shape)
        # caffe_in[0].tofile('caffe_in.bin')

        # forward
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]

        # [DEBUG] dump output to file
        # print("predictions[0].shape", predictions[0].shape)
        # predictions[0].tofile('caffe_out.bin')

        # dump blobs to file
        # this dump all blobs even if some of them are `in_place` blobs
        if dump_blobs is not None:
            print("Save Blobs: ", dump_blobs)
            blobs_dict = {}

            blobs_dict['raw_data'] = input_
            # for name, blob in self.blobs.iteritems():
            #     blobs_dict[name] = blob.data
            for name, layer in self.layer_dict.items():
                print("layer : " + str(name))
                print("  type = " + str(layer.type))
                print("  top -> " + str(self.top_names[name]))
                if layer.type == "Split":
                    print("  skip Split")
                    continue
                if layer.type == "Slice":
                    print(" skip Slice")
                    continue
                assert(len(self.top_names[name]) == 1)
                if layer.type == "Input":
                    blobs_dict[name] = caffe_in
                    continue
                #out = self.forward(None, prev_name, name, **{prev_name: prev_data})
                out = self.forward(None, None, name, **{self.inputs[0]: caffe_in})
                blobs_dict[name] = out[self.top_names[name][0]].copy()
            np.savez(dump_blobs, **blobs_dict)

        # dump weight to file
        if dump_weights is not None:
            print("Save Weights:", dump_weights)
            weights_dict = {}
            for name, param in self.params.items():
                for i in range(len(param)):
                    weights_dict[name + "_" + str(i)] = param[i].data
            np.savez(dump_weights, **weights_dict)

        return predictions

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )
    parser.add_argument(
        "--net_input_dims",
        default='224,224',
        help="'height,width' dimensions of network input spatial dimensions."
    )
    parser.add_argument(
        "--image_resize_dims",
        default='256,256',
        help="To resize to this size first, then crop to net_input_dims."
    )
    parser.add_argument(
        "--mean_file",
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--mean",
        help="Per Channel image mean values"
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--label_file",
        help="Labels file"
    )
    parser.add_argument(
        "--dump_blobs",
        help="Dump all blobs into a file in npz format"
    )
    parser.add_argument(
        "--dump_weights",
        help="Dump all weights into a file in npz format"
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=1,
        help="Set batch size"
    )
    parser.add_argument(
        "--force_input",
        help="Force the input blob data, in npy format"
    )
    args = parser.parse_args()

    image_resize_dims = [int(s) for s in args.image_resize_dims.split(',')]
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    image_resize_dims = [ max(x,y) for (x,y) in zip(image_resize_dims, net_input_dims)]

    mean, channel_swap = None, None
    if args.mean:
        mean = np.array([float(s) for s in args.mean.split(',')], dtype=np.float32)
    else:
        if args.mean_file:
            mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    # Make classifier.
    classifier = My_Classifier(args.model_def, args.pretrained_model,
            image_dims=image_resize_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap, batch_size=args.batch_size)

    # Load image file.
    args.input_file = os.path.expanduser(args.input_file)
    print("Loading file: %s" % args.input_file)

    input_x = [caffe.io.load_image(args.input_file)]
    inputs = input_x
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, input_x, axis=0)
    # Classify.
    start = time.time()
    predictions = classifier.predict(inputs,
                                     args.dump_blobs, args.dump_weights,
                                     args.force_input)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, predictions)

    # Print
    for ix, in_ in enumerate(inputs):
      print("batch : ", ix)
      print(predictions[ix].argmax())
      if args.label_file:
         labels = np.loadtxt(args.label_file, str, delimiter='\t')
         top_k = predictions[ix].flatten().argsort()[-1:-6:-1]
         print(labels[top_k])
         print(top_k)
         prob = np.squeeze(predictions[ix].flatten())
         idx = np.argsort(-prob)
         for i in range(5):
             label = idx[i]
             print('%d - %.2f - %s' % (idx[i], prob[label], labels[label]))

if __name__ == '__main__':
    main(sys.argv)
