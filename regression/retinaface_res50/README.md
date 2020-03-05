# retinaface_res50

## Model

link

- From insightface project, `https://github.com/deepinsight/insightface/tree/master/RetinaFace`

  MXNet model

  R50, `https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ` or `https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0`

  mnet, `https://github.com/deepinsight/insightface/issues/669` or `https://drive.google.com/drive/folders/1OTXuAUdkLVaf78iz63D1uqGLZi4LbPeL?usp=sharing`

- Convert to Caffe, `https://github.com/clancylian/retinaface`

  With MXnet2Caffe tool, `https://github.com/cypw/MXNet2Caffe`

- Or Download Caffe Model directly, `https://github.com/Charrin/RetinaFace-Cpp`

  R50, `https://drive.google.com/drive/folders/1hA5x3jCYFdja3PXLl9EcmucipRmVAj3W?usp=sharing`

  mnet, `https://github.com/Charrin/RetinaFace-Cpp/tree/master/convert_models/mnet`

- We use the Download one for now (The Charrin one)

## Dataset

- widerface: `http://shuoyang1213.me/WIDERFACE/index.html`

  val: `https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing`

  test: `https://drive.google.com/file/d/0B6eKvaijfFUDbW4tdGpaYjgzZkU/view?usp=sharing`

  anno: `http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip`

  get the val and anno

## Performance Results

- cv1835 (DDR3)
  - retinaface mobilenet 320x320  1.69 ms, 591.03 fps
  - retinaface resnet50 320x320   35.19 ms, 28.41 fps
  - retinaface resnet50 600x600   112.7802 ms, 8.86 fps

## Accuracy Results

- widerface easy/medium/hard
  - FP32
    - retinaface_res50     600x600   0.904/0.879/0.619
    - retinaface_res50     320x320   0.853/0.756/0.345
    - retinaface_mobilenet 600x600   0.817/0.757/0.479   ****
    - retinaface_mobilenet 320x320   0.709/0.585/0.257

  - INT8
    - retinaface_res50     600x600  MLIR   kld   0.812/0.704/0.303
    - retinaface_res50     320x320  bmtap2 tune  0.768/0.657/0.294
    - retinaface_mobilenet 320x320  MLIR   kld   0.480/0.303/0.127
    - retinaface_mobilenet 320x320  bmtap2 tune  0.677/0.553/0.242