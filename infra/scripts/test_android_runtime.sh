#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

# Model download server setting
if [[ -z "${MODELFILE_SERVER}" ]]; then
  echo "[ERROR] Model file server is not set"
  echo "        Need to download model file for test"
  exit 1
else
  echo "Model Server: ${MODELFILE_SERVER}"
fi

$ROOT_PATH/Product/out/test/models/run_test.sh --download=on --run=off
$ROOT_PATH/Product/out/test/models/run_test.sh --download=on --run=off \
  --configdir=$ROOT_PATH/Product/out/test/models/nnfw_api_gtest \
  --cachedir=$ROOT_PATH/Product/out/unittest_standalone/nnfw_api_gtest_models

N=`adb devices 2>/dev/null | wc -l`

# exit if no device found
if [[ $N -le 2 ]]; then
    echo "No device found."
    exit 1;
fi

NUM_DEV=$(($N-2))
echo "device list"
DEVICE_LIST=`adb devices 2>/dev/null`
echo "$DEVICE_LIST" | tail -n"$NUM_DEV"

if [ -z "$SERIAL" ]; then
    SERIAL=`echo "$DEVICE_LIST" | tail -n1 | awk '{print $1}'`
fi
ADB_CMD="adb -s $SERIAL "

# root on, remount as rw
$ADB_CMD root on
$ADB_CMD shell mount -o rw,remount /

$ADB_CMD shell mkdir -p /data/local/tmp/report/benchmark
$ADB_CMD push $ROOT_PATH/tests /data/local/tmp/.
$ADB_CMD push $ROOT_PATH/Product/aarch64-android.release/out /data/local/tmp/Product/.

$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/Product/lib sh /data/local/tmp/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/Product/bin/tflite_loader_test_tool \
                                                        --reportdir=/data/local/tmp/report \
                                                        --tapname=tflite_loader
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/Product/lib sh /data/local/tmp/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/Product/bin/nnapi_test \
                                                        --reportdir=/data/local/tmp/report \
                                                        --tapname=nnapi_test
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/Product/lib USE_NNAPI=1 sh /data/local/tmp/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/Product/bin/tflite_run \
                                                        --reportdir=/data/local/tmp/report \
                                                        --tapname=tflite_run

# $ADB_CMD shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp/Product/lib sh /data/local/tmp/tests/scripts/test_scheduler_with_profiling_android.sh'

mkdir -p $ROOT_PATH/report

$ADB_CMD pull /data/local/tmp/report $ROOT_PATH/report/android