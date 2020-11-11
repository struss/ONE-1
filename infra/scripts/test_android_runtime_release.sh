#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

: ${DEVICE:="none"}
TEST_ARCH="aarch64"
TEST_OS="android"
TEST_PLATFORM="$TEST_ARCH-$TEST_OS"
EXECUTORS=("Linear" "Dataflow" "Parallel")

# Model download server setting
if [[ -z "${MODELFILE_SERVER}" ]]; then
  echo "[ERROR] Model file server is not set"
  echo "        Need to download model file for test"
  exit 1
else
  echo "Model Server: ${MODELFILE_SERVER}"
fi

apt-get update && apt-get install -y curl

BACKENDS=( "acl_neon" "cpu" "acl_cl" )

$ROOT_PATH/tests/scripts/models/run_test.sh --download=on --run=off
$ROOT_PATH/Product/aarch64-android.release/out/test/models/run_test.sh --download=on --run=off \
  --configdir=$ROOT_PATH/Product/aarch64-android.release/out/test/models/nnfw_api_gtest \
  --cachedir=$ROOT_PATH/Product/aarch64-android.release/out/unittest_standalone/nnfw_api_gtest_models

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
echo $SERIAL
ADB_CMD="adb -s $SERIAL "

# root on, remount as rw
$ADB_CMD root on
$ADB_CMD shell mount -o rw,remount /

$ADB_CMD shell rm -rf /data/local/tmp/onert_android
$ADB_CMD shell rm -rf /data/local/tmp/TestCompilationCaching*
$ADB_CMD shell mkdir -p /data/local/tmp/onert_android/report
$ADB_CMD push $ROOT_PATH/tests /data/local/tmp/onert_android/.
$ADB_CMD push $ROOT_PATH/Product/aarch64-android.release/out /data/local/tmp/onert_android/Product/.


TESTLIST=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/tflite_loader_list.${TEST_ARCH}.txt")
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib BACKENDS=acl_cl sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/tflite_loader_test_tool \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=tflite_loader.tap ${TESTLIST:-}
UNION_MODELLIST_PREFIX="${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.${TEST_ARCH}"
sort $UNION_MODELLIST_PREFIX.${BACKENDS[0]}.txt > $UNION_MODELLIST_PREFIX.intersect.txt
for BACKEND in "${BACKENDS[@]}";
do
  comm -12 <(sort $UNION_MODELLIST_PREFIX.intersect.txt) <(sort $UNION_MODELLIST_PREFIX.$BACKEND.txt) > $UNION_MODELLIST_PREFIX.intersect.next.txt
  mv $UNION_MODELLIST_PREFIX.intersect.next.txt $UNION_MODELLIST_PREFIX.intersect.txt
  for EXECUTOR in "${EXECUTORS[@]}";
  do
    MODELLIST=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.${TEST_ARCH}.${BACKEND}.txt")
    SKIPLIST=$(grep -v '#' "${ROOT_PATH}/Product/aarch64-android.release/out/unittest/nnapi_gtest.skip.${TEST_PLATFORM}.${BACKEND}" | tr '\n' ':')
    $ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKENDS=$BACKEND /data/local/tmp/onert_android/Product/unittest/nnapi_gtest \
                                                            --gtest_output=xml:/data/local/tmp/onert_android/report/nnapi_gtest_${BACKEND}_${EXECUTOR}.xml \
                                                            --gtest_filter=-${SKIPLIST}
    $ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKENDS=$BACKEND sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                            --driverbin=/data/local/tmp/onert_android/Product/bin/nnapi_test \
                                                            --reportdir=/data/local/tmp/onert_android/report \
                                                            --tapname=nnapi_test_${BACKEND}_${EXECUTOR}.tap ${MODELLIST:-}
  done
done

MODELLIST_INTERP=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.noarch.interp.txt")
SKIPLIST_INTERP=$(grep -v '#' "${ROOT_PATH}/Product/aarch64-android.release/out/unittest/nnapi_gtest.skip.noarch.interp" | tr '\n' ':')
for EXECUTOR in "${EXECUTORS[@]}";
do
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKENDS=$BACKEND /data/local/tmp/onert_android/Product/unittest/nnapi_gtest \
                                                            --gtest_output=xml:/data/local/tmp/onert_android/report/nnapi_gtest_interp_$EXECUTOR.xml \
                                                            --gtest_filter=-$SKIPLIST_INTERP
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKENDS="" DISABLE_COMPILE=1 sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/nnapi_test \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=nnapi_test_interp_$EXECUTOR.tap ${MODELLIST_INTERP:-}
done

MODELLIST=$(cat "${UNION_MODELLIST_PREFIX}.intersect.txt")
for EXECUTOR in "${EXECUTORS[@]}";
do
# $ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKENDS=$BACKEND /data/local/tmp/onert_android/Product/unittest/nnapi_gtest \
#                                                             --gtest_output=xml:/data/local/tmp/onert_android/report/nnapi_gtest_${BACKEND}_${EXECUTOR}.xml \
#                                                             --gtest_filter=-${SKIPLIST}
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib OP_BACKEND_Conv2D="cpu" OP_BACKEND_MaxPool2D="acl_cl" OP_BACKEND_AvgPool2D="acl_neon" ACL_LAYOUT="NCHW" EXECUTOR=$EXECUTOR BACKENDS="acl_cl\;acl_neon\;cpu" sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/nnapi_test \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=nnapi_test_mixed_$EXECUTOR.tap ${MODELLIST:-}
done
# $ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib USE_NNAPI=1 sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
#                                                         --driverbin=/data/local/tmp/onert_android/Product/bin/tflite_run \
#                                                         --reportdir=/data/local/tmp/onert_android/report \
#                                                         --tapname=tflite_run.tap

# This is test for profiling.
# $ADB_CMD shell mkdir -p /data/local/tmp/onert_android/report/benchmark
# $ADB_CMD shell 'cd /data/local/tmp/onert_android && LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib sh /data/local/tmp/onert_android/tests/scripts/test_scheduler_with_profiling_android.sh'

rm -rf $ROOT_PATH/report
mkdir -p $ROOT_PATH/report


$ADB_CMD pull /data/local/tmp/onert_android/report $ROOT_PATH/report/$DEVICE
