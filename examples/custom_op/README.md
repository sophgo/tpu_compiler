自定义OP实现说明

## sample验证方法

1) 与cvitek_mlir放在同级目录，进入cvitek_mlir后`source cvitek_envs.sh`
2) 进入该sample目录，执行`build.sh`

## 添加自定义OP步骤

1) 在model目录下caffe prototxt中用Python layer定义新OP, 参考MyAdd和MyMul
2) 在model目录下mymodel.py中`convert_python_op`接口实现新OP到mlir文件的转换
3) 在code/python目录添加caffe layer的实现
4) 在code/目录下添加op目录，参考`my_add`和`my_mul`实现接口

