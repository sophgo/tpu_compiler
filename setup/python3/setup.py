#!/usr/bin/python3
import os, sys
import setuptools
import glob

install_requires=[
    'numpy>=1.18.0',
    'opencv-python>=3.4.0.14',
    'protobuf==3.11.3',
    'scikit-image==0.14.5',
    'onnx==1.6.0',
    'onnxruntime==1.1.2',
    'torch==1.4.0',
    'torchvision==0.5.0',
    'termcolor==1.1.0',
    'PyYAML==5.3.1',
]
python_version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

mlir_install_path = os.environ.get('INSTALL_PATH')
install_lib = '{}/lib'.format(mlir_install_path)
install_bin = '{}/bin'.format(mlir_install_path)
install_py_lib = '{}/python'.format(mlir_install_path)

so_lib = [ x for x in glob.iglob('{}/*.so'.format(install_lib))]
a_lib = [ x for x in glob.iglob('{}/*.a'.format(install_lib))]
py_so_lib = [ x for x in glob.iglob('{}/*.so'.format(install_py_lib))]
cvi_bin = [x for x in glob.iglob('{}/*'.format(install_bin))]


file_path = os.path.dirname(os.path.abspath(__file__))
print("setup.py in {}".format(file_path))
root_path = os.path.join(file_path, "../../")
print("Now root path is {}".format(root_path))

os.chdir(root_path)
packages = setuptools.find_packages(where='{}/python'.format(root_path)) + setuptools.find_packages(where='./third_party/caffe/python')
setuptools.setup(
    name='CVI_toolkit',
    version='0.5.0',
    keywords='cvi toolkit',
    description='CVI tool python packge',
    author='sam.zheng',
    author_email='sam.zheng@wisecore.com.tw',
    packages=packages,
    package_dir={
        '': 'python',
        'caffe': "third_party/caffe/python/caffe"
    },
    package_data ={
        'caffe': ["*.so"]
    },
    data_files=[
        ('lib', so_lib),
        ('lib', a_lib),
        ('lib/python{}'.format(python_version), py_so_lib),
        ('bin', cvi_bin),
    ],
    install_requires=install_requires,
    entry_points = {
              'console_scripts': [
                  'cvi_npz_tool=cvi_toolkit.cvi_npz_tool:main',
                  'cvi_model_convert=cvi_toolkit.cvi_model_convert:main',
                  'cvi_calibration_tool=cvi_toolkit.calibration:main',
                  'cvi_nn_converter_tool=cvi_toolkit.cvi_nn_converter:main'
              ],
          },

)