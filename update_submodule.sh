#!/bin/bash
repo_list=(
"externals/cmodel"
"externals/cvibuilder"
"externals/cvikernel"
"externals/cvimath"
"externals/cviruntime"
"externals/profiling"
"third_party/cnpy"
"third_party/flatbuffers"
"third_party/opencv"
"third_party/pybind11"
"third_party/systemc-2.3.3"
)

echo "" > /tmp/sdk_update.txt
for repo in ${repo_list[@]}
do
    pushd $repo
    if [[ $repo == "third_party/opencv" ]]; then
       git checkout tpu
    elif [[ $repo == "third_party/cnpy" ]]; then
       git checkout tpu
    elif [[ $repo == "third_party/pybind11" ]]; then
       git checkout tpu
    elif [[ $repo == "third_party/caffe" ]]; then
       git checkout tpu_master
    else
       git checkout master
    fi
    git pull
    if [[ "$?" -ne "0" ]]; then
       echo "ERROR...."
       echo "$repo ERROR" >> /tmp/sdk_update.txt
    else
       echo "OK...."
       echo "$repo OK" >> /tmp/sdk_update.txt
    fi
    popd
done

pushd third_party/caffe
git checkout tpu_master
git pull
if [[ "$?" -ne "0" ]]; then
   echo "ERROR...."
   echo "third_party/caffe ERROR" >> /tmp/sdk_update.txt
else
   echo "OK...."
   echo "third_party/caffe OK" >> /tmp/sdk_update.txt
fi
popd

cat /tmp/sdk_update.txt
