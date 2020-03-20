#!/usr/bin/python
import os
import setuptools


install_requires=[
    'enum34==1.1.10',
    'numpy==1.16.6',
    'opencv-python==4.2.0.32',
    'protobuf==3.11.3',
    'scikit-image==0.14.5'
]




setuptools.setup(
    name='CVI_toolkit',
    version='0.5.0',
    keywords='cvi toolkit',
    description='CVI tool python packge',
    author='sam.zheng',
    author_email='sam.zheng@wisecore.com.tw',
    packages=setuptools.find_packages(where='python'),
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