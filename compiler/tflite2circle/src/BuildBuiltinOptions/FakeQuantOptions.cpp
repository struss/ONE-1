/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "FillOptions.h"

namespace tflite2circle
{

flatbuffers::Offset<circle::FakeQuantOptions>
build_circle_FakeQuantOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_FakeQuantOptions();
  assert(tflite_builtin_options);
  circle::FakeQuantOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_min(tflite_builtin_options->min());
  builtin_options_builder.add_max(tflite_builtin_options->max());
  builtin_options_builder.add_num_bits(tflite_builtin_options->num_bits());
  builtin_options_builder.add_narrow_range(tflite_builtin_options->narrow_range());
  return builtin_options_builder.Finish();
}

} // namespace tflite2circle
