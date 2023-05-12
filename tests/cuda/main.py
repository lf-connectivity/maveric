# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

print(f"PyTorch CUDA version?: {torch.version.cuda}")
print(f"PyTorch CUDA is_available?: {torch.cuda.is_available()}")
print(f"PyTorch cuDNN enabled?: {torch.backends.cudnn.enabled}")

torch.zeros(1).cuda()
