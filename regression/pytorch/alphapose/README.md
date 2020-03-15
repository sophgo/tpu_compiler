# AlphaPose

# Model Path

  `https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md`

  Download the `Fast Pose`
    - `https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn`, as `alpha_pose_res50_256x192.pth`
    - `https://raw.githubusercontent.com/MVIG-SJTU/AlphaPose/master/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml`

  ```sh
  $ git clone https://github.com/MVIG-SJTU/AlphaPose.git
  $ cd AlphaPose
  ```

  ```sh
  $ bash Miniconda3-latest-Linux-x86_64.sh
  $ conda create -n mytorch python=3.6 -y
  $ conda activate mytorch
  $ conda install pytorch torchvision
  $ export PATH=/usr/local/cuda/bin/:$PATH
  $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
  $ python setup.py build develop --user
  ```

  copy alpha_pose_res50_256x192.pth to scripts
  copy 256x192_res50_lr1e-3_1x.yaml to scripts
  copy pose.npz to alphapose root dir

  ```
  $ conda install pillow=6.1
  $ python3 convert_to_onnx.py
  Finish!
  ```
  Got alphapose.onnx (162572715 bytes)

# Convert to Onnx

In AlphaPose Repo create python file create_onnx.py, in release repo only accept py3

please use python3 create_onnx

```
import numpy as np
import torch
from alphapose.models import builder
from alphapose.utils.config import update_config

cfg_path = 'scripts/256x192_res50_lr1e-3_1x.yaml'
weight = 'scripts/fast_res50_256x192.pth'
model_path = 'alpha_pose_res50_256x192.pth'
onnx_model_name = 'alphapose.onnx'

if __name__ == "__main__":
    cfg = update_config(cfg_path)
    input_npz = np.load('pose.npz')
    input = input_npz['input']
    input = torch.from_numpy(input)
    device = torch.device('cpu')
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(weight, map_location=device))
    torch.onnx.export(pose_model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  onnx_model_name,          # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
    print("Finish!")

```

get a onnx model

However, use python2 to run convert.py (convert to mlir) for now

# How to do calibration

1. Put all human detect npz files to one folder(e.x. data)
2. Readlink -f data/* > input.txt
3. Run calibration

python run_calibration.py --model_name=alpha_pose alphapose_opt.mlir input.txt  --input_num=20



