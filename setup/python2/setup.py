#!/usr/bin/python
import os
import setuptools


install_requires=[
    'enum34==1.1.10',
    'numpy==1.16.6',
    'opencv-python>=3.4.0.14',
    'protobuf==3.11.3',
    'scikit-image==0.14.5'
]

file_path = os.path.dirname(os.path.abspath(__file__))
print("setup.py in {}".format(file_path))
root_path = os.path.join(file_path, "../../")
print("Now root path is {}".format(root_path))

os.chdir(root_path)

setuptools.setup(
    name='CVI_toolkit',
    version='0.5.0',
    keywords='cvi toolkit',
    description='CVI tool python packge',
    author='sam.zheng',
    author_email='sam.zheng@wisecore.com.tw',
    packages=setuptools.find_packages(where='{}/python'.format(root_path)),
    package_dir={
        '': 'python',
    },
    install_requires=install_requires,
    entry_points = {
              'console_scripts': [
                  'cvi_npz_tool=CVI_toolkit.cvi_npz_tool:main',
                  'cvi_model_convert=CVI_toolkit.cvi_model_convert.py:main'
              ],
          },

)