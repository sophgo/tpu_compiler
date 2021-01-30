#!/bin/bash
# set -e
# set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WORKING_PATH=${WORKING_PATH:-$DIR}
export MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-8}

run_gen_cvimodel()
{
  local net=$1
  echo "generate cvimodel for $net"
  generate_cvimodel.sh $net > $net.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    echo "$net cvimodel generated FAILED"
    return 1
  else
    rm -f $net.log
    return 0
  fi
}
export -f run_gen_cvimodel

run_gen_cvimodel_all()
{
  local err=0
  for net in ${all_net_list[@]}
  do
    run_gen_cvimodel $net
    if [ "$?" -ne 0 ]; then
      return 1
    fi
  done
  return 0
}

run_gen_cvimodel_all_parallel()
{
  echo "MAX_PARALLEL_JOBS: ${MAX_PARALLEL_JOBS}"

  rm -f models.txt
  for net in ${all_net_list[@]}
  do
    echo "run_gen_cvimodel $net" >> models.txt
  done
  cat models.txt
  parallel -j${MAX_PARALLEL_JOBS} < models.txt
  rm models.txt
  return $?
}

# default run in parallel
if [ -z "$RUN_IN_PARALLEL" ]; then
  export RUN_IN_PARALLEL=1
fi

all_net_list=()

if [ -z $model_list_file ]; then
  model_list_file=$DIR/generic/model_list.txt
fi
while read net bs1 bs4 acc bs1_ext bs4_ext acc_ext
do
  [[ $net =~ ^#.* ]] && continue
  all_net_list+=($net)
done < ${model_list_file}

err=0
mkdir -p $WORKING_PATH/cvimodel_release
pushd $WORKING_PATH/cvimodel_release

if [ "$RUN_IN_PARALLEL" -eq 0 ]; then
  run_gen_cvimodel_all
  if [ "$?" -ne 0 ]; then
    err=1
  fi
else
  run_gen_cvimodel_all_parallel
  if [ "$?" -ne 0 ]; then
    err=1
  fi
fi

popd

exit $err
