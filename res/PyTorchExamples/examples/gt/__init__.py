import torch
import torch.nn as nn


# model
class net_gt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.gt(inputs[0], inputs[1])


_model_ = net_gt()

# dummy input for onnx generation
_dummy_ = [torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)]
