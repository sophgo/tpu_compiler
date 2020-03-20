# arcface

arcface\_res50  prototxt and caffemodel convert from mxnet model:

- `https://github.com/deepinsight/insightface/wiki/Model-Zoo`

- `https://www.dropbox.com/s/ou8v3c307vyzawc/model-r50-arcface-ms1m-refine-v1.zip?dl=0`

The original model is mxnet, use mxnet2caffe tool to convert

```sh
$ python json2prototxt.py \
    --mx-json /data/models/face_recognition/arcface/mxnet/model-r50-am-lfw/model-symbol.json \
    --cf-prototxt arcface_res50.prototxt
$ python mxnet2caffe.py \
    --mx-model /data/models/face_recognition/arcface/mxnet/model-r50-am-lfw/model \
    --cf-prototxt arcface_res50.prototxt \
    --cf-model arcface_res50.caffemodel
```
