# MLIR UI Convert
we implement GUI for friendly used
the low level command just call `convert_model.sh`

the success convert message as ![image](https://cvitekcn-my.sharepoint.com/personal/arvin_chou_cvitek_com/Documents/191114/a/Screenshot%20from%202020-10-06%2021-15-03.png)

and the fail case as ![image](https://cvitekcn-my.sharepoint.com/personal/arvin_chou_cvitek_com/Documents/191114/a/Screenshot%20from%202020-10-06%2021-15-50.png)

# how to use
```sh
# # you are at MLIR root
# pip3 install -r requirements.txt 
# cd python/cvi_toolkit/convert_ui
# python3 convert.py
```

## v0.1
1. no foolproof and error handling
2. implemnted by pygubu, tkinter
3. only test under caffe squeezenet
