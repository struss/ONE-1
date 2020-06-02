/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <jni.h>

#include "onert-native-internal.h"

namespace jni_helper
{

jboolean verifyHandle(jlong handle);
jboolean getTensorParams(JNIEnv *env, jint jindex, jint jtype, jobject jbuf, jint jbufsize,
                         jni::TensorParams &params);
jboolean getTensorParams(jint jindex, jint jtype, jlong handle, jni::TensorParams &params);
jboolean getLayoutParams(jint jindex, jint jlayout, jni::LayoutParams &params);
jboolean setTensorInfoToJava(JNIEnv *env, const nnfw_tensorinfo &tensor_info, jobject jinfo);
jboolean getInputTensorInfo(jlong handle, jint jindex, jni::TensorInfo &info);
jboolean getOutputTensorInfo(jlong handle, jint jindex, jni::TensorInfo &info);

} // namespace jni_helper
