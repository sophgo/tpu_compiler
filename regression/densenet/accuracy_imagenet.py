import numpy as np
import os
import caffe
import pymlir

from argparse import ArgumentParser

# for densenet
# int8 per layer : calibration 1000, val 4270 pics, accuracy: 0.657
# int8 per channel : calibration 100, val 4000 pics, accuracy: 0.71 ~0.72
#                    calibration 1000, val 20000pics, accuracy: 0.70
# int8 multipiler: calibration 100, val 12700 pics, accuracy: 0.70 (10 hours)

#  run mlir model:
#  python ../accuracy_imagenet.py --mlir densenet_quant_int8_per_layer.mlir \
#         --dataset ~/dataset/imagenet/img_val_extracted/val/ 
#         --label ~/dataset/imagenet/img_val_extracted/val.txt 
#         --mean 103.94,116.78,123.68 
#         --input_scale 0.017
#         --input_shape 1,3,224,224

# run caffe model:
#  python ../accuracy_imagenet.py 
#         --proto ~/work/models/imagenet/densenet/caffe/densenet121_deploy.prototxt
#         --model ~/work/models/imagenet/densenet/caffe/densenet121.caffemodel 
#         --dataset ~/dataset/imagenet/img_val_extracted/val/ 
#         --label ~/dataset/imagenet/img_val_extracted/val.txt 
#         --mean 103.94,116.78,123.68 
#         --input_scale 0.017

# place images under img_val_extracted/val and label list in val.txt


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--proto", help="prototxt path")
    parser.add_argument("--model", help="caffe model")
    parser.add_argument("--mlir", help="mlir model")
    parser.add_argument("--dataset", help="eval imagenet dataset path")
    parser.add_argument("--label_file", help="imagenet label file")
    parser.add_argument("--mean_file", help="dataset mean file")
    parser.add_argument("--mean", help="mean array")
    parser.add_argument("--input_scale", type=float, help="input scale, default is 1.0")
    parser.add_argument("--input_shape", help="input shape for mlir")

    args = parser.parse_args()

    mean = None
    input_scale = 1.0
    if (args.mean):
        mean = np.array( [float(s) for s in args.mean.split(',')], dtype = np.float32)
    else:
        if (args.mean_file):
            mean = np.load(args.mean_file)
    if (args.input_scale):
        input_scale = args.input_scale
    
    caffenet = None
    mlir = None
    input_shape = None
    if (args.proto):
        caffenet = caffe.Net(args.proto, args.model, caffe.TEST)
        input_shape = caffenet.blobs['data'].data.shape

    if (args.mlir):
        mlir = pymlir.module()
        mlir.load(args.mlir)
        if (args.input_shape):
            shape = [int(s) for s in args.input_shape.split(',')]
            input_shape = tuple(shape)
        else:
            print "need specify --input_shape xx,xx,xx,xx for mlir"
            exit(-1)
    
    print "shape:", input_shape

    if (args.proto and args.mlir):
        print "can not run mlir and caffe at one time"
        exit(-1)
    
    transformer = caffe.io.Transformer({'data': input_shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_mean('data', mean)
    transformer.set_input_scale('data', input_scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    with open(args.label_file, 'r') as f:
        images_path = f.readlines()

    count = 0
    top1_correct = 0.0
    #top5_correct = 0.0
    total_count = len(images_path)
    while count <= total_count:
        image_name = images_path[count].split(' ')[0]
        image_path = os.path.join(args.dataset, image_name)
        img_data = caffe.io.load_image(image_path)
        img_data = transformer.preprocess('data', img_data)
        print(img_data.shape)

        if (caffenet):
            caffenet.blobs['data'].data[0][...] = img_data
            out = caffenet.forward()
            out = out['fc6']
        elif (mlir):
            img_data = np.expand_dims(img_data, axis=0)
            out = mlir.run(img_data)
            out = out.values()[0]
            out = np.reshape(out, (out.shape[0], out.shape[1]))
        
        top1 = np.argsort(out.flatten())[-1]
        label = int(images_path[count].split(' ')[1])
        
        if top1 == label:
            top1_correct = top1_correct + 1

        count = count + 1
        if count % 10 == 0:
            print("correct top1 count: {}".format(top1_correct))
            print("current eval count: {}".format(count))
            print("accuracy: {}".format(top1_correct / count))

    accuracy = top1_correct / float(total_count)
    print("accuracy = {}%".format(accuracy))

