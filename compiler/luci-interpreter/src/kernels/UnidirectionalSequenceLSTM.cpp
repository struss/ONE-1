/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/UnidirectionalSequenceLSTM.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/lstm_eval.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

UnidirectionalSequenceLSTM::UnidirectionalSequenceLSTM(
    const Tensor *input, const Tensor *input_to_input_weights,
    const Tensor *input_to_forget_weights, const Tensor *input_to_cell_weights,
    const Tensor *input_to_output_weights, const Tensor *recurrent_to_input_weights,
    const Tensor *recurrent_to_forget_weights, const Tensor *recurrent_to_cell_weights,
    const Tensor *recurrent_to_output_weights, const Tensor *cell_to_input_weights,
    const Tensor *cell_to_forget_weights, const Tensor *cell_to_output_weights,
    const Tensor *input_gate_bias, const Tensor *forget_gate_bias, const Tensor *cell_gate_bias,
    const Tensor *output_gate_bias, const Tensor *projection_weights, const Tensor *projection_bias,
    const Tensor *output_state, const Tensor *cell_state,
    const Tensor *input_layer_norm_coefficients, const Tensor *forget_layer_norm_coefficients,
    const Tensor *cell_layer_norm_coefficients, const Tensor *output_layer_norm_coefficients,
    Tensor *output, const UnidirectionalSequenceLSTMParams &params)
    : KernelWithParams<UnidirectionalSequenceLSTMParams>({input,
                                                          input_to_input_weights,
                                                          input_to_forget_weights,
                                                          input_to_cell_weights,
                                                          input_to_output_weights,
                                                          recurrent_to_input_weights,
                                                          recurrent_to_forget_weights,
                                                          recurrent_to_cell_weights,
                                                          recurrent_to_output_weights,
                                                          cell_to_input_weights,
                                                          cell_to_forget_weights,
                                                          cell_to_output_weights,
                                                          input_gate_bias,
                                                          forget_gate_bias,
                                                          cell_gate_bias,
                                                          output_gate_bias,
                                                          projection_weights,
                                                          projection_bias,
                                                          output_state,
                                                          cell_state,
                                                          input_layer_norm_coefficients,
                                                          forget_layer_norm_coefficients,
                                                          cell_layer_norm_coefficients,
                                                          output_layer_norm_coefficients},
                                                         {output}, params)
{
}

void UnidirectionalSequenceLSTM::configure()
{
  _is_layer_norm_lstm = true;
  if (input_layer_norm_coefficients() == nullptr && forget_layer_norm_coefficients() == nullptr &&
      cell_layer_norm_coefficients() == nullptr && output_layer_norm_coefficients() == nullptr)
  {
    _is_layer_norm_lstm = false;
  }
  else
  {
    if (forget_layer_norm_coefficients() == nullptr)
      _is_layer_norm_lstm = false;
  }
  LUCI_INTERPRETER_CHECK(input()->shape().num_dims() > 1);
  bool time_major = params().time_major;
  _scratch_tensor =
      std::make_unique<Tensor>(DataType::S32, output()->shape(), AffineQuantization{}, "");
}

void UnidirectionalSequenceLSTM::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void UnidirectionalSequenceLSTM::evalFloat() const {}

} // namespace kernels
} // namespace luci_interpreter
