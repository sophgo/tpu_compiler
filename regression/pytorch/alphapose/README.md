# AlphaPose

# Model Path

```https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md```

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

# How to do calibration

1. Put all human detect npz files to one folder(e.x. data)
2. Readlink -f data/* > input.txt
3. Run calibration

python run_calibration.py --model_name=alpha_pose alphapose_opt.mlir input.txt  --input_num=20



