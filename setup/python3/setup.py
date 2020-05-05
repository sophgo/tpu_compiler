#!/usr/bin/python3
import os, sys
import setuptools
import glob
import subprocess

IS_PY3 = sys.version_info >= (3,0)
if not IS_PY3:
    print("python version {} < 3".format(sys.version_info))
    exit(-1)

with open("requirements.txt", "r") as req:
    package = req.readlines()
install_requires=[x.strip() for x in package]

class CleanCommand(setuptools.Command):
    """Custom clean command to tidy up the project root."""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -vrf ./build ./dist ./python/*.egg-info')

def get_git_revision_short_hash():
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return commit.decode("utf-8").strip()


python_version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)

mlir_install_path = os.environ.get('INSTALL_PATH')
install_lib = '{}/lib'.format(mlir_install_path)
install_bin = '{}/bin'.format(mlir_install_path)
install_py_lib = '{}/python'.format(mlir_install_path)

so_lib = [ x for x in glob.iglob('{}/*.so'.format(install_lib))]
a_lib = [ x for x in glob.iglob('{}/*.a'.format(install_lib))]
py_so_lib = [ x for x in glob.iglob('{}/*.so'.format(install_py_lib))]
cvi_bin = [x for x in glob.iglob('{}/*'.format(install_bin))]


caffe_path = "{}/caffe".format(mlir_install_path)
caffe_lib = [ x for x in glob.iglob('{}/lib/**'.format(caffe_path))]

# mkldnn
mkldnn_path = "{}/mkldnn".format(mlir_install_path)
mkldnn_lib = [ x for x in glob.iglob('{}/lib/*.so*'.format(mkldnn_path))]

file_path = os.path.dirname(os.path.abspath(__file__))
print("setup.py in {}".format(file_path))
root_path = os.path.join(file_path, "../../")
print("Now root path is {}".format(root_path))

os.chdir(root_path)
packages = setuptools.find_packages(where='{}/python'.format(root_path))
packages.extend(setuptools.find_packages(where='./third_party/caffe/python'))
packages.extend(setuptools.find_packages(where='./third_party/flatbuffers/python'))

# cpu op, generated runtime, we hardcore here
packages.extend(['cvi', 'cvi.cpu_op', 'cvi.model'])
# tflite
packages.extend(['tflite'])
setuptools.setup(
    name='CVI_toolkit',
    version=get_git_revision_short_hash(),
    keywords='cvi toolkit',
    description='CVI tool python packge',
    author='sam.zheng',
    author_email='sam.zheng@wisecore.com.tw',
    packages=packages,
    package_dir={
        '': 'python',
        'caffe': 'third_party/caffe/python/caffe',
        'flatbuffers': 'third_party/flatbuffers/python/flatbuffers',
        'cvi': '{}/python/cvi'.format(mlir_install_path),
        'tflite':'{}/python/tflite'.format(mlir_install_path),
    },
    package_data ={
        'caffe': ["*.so"]
    },
    data_files=[
        ('lib/python{}'.format(python_version), so_lib),
        ('lib/python{}'.format(python_version), py_so_lib),
        ('lib/python{}'.format(python_version), mkldnn_lib),
        ('lib', mkldnn_lib),
        ('lib', so_lib),
        ('lib', caffe_lib),
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
    cmdclass={
        'clean': CleanCommand,
    },

)