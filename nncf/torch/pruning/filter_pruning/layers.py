"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np
import torch
from torch import nn

from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.common.graph import NNCFNodeName
from nncf.torch.utils import is_tracing_state, no_jit_trace


@COMPRESSION_MODULES.register()
class FilterPruningBlock(nn.Module):
    def __init__(self, size, dim=0):
        super().__init__()
        self.register_buffer("_binary_filter_pruning_mask", torch.ones(size))
        self.mask_applying_dim = dim

    @property
    def binary_filter_pruning_mask(self):
        return self._binary_filter_pruning_mask

    @binary_filter_pruning_mask.setter
    def binary_filter_pruning_mask(self, mask):
        with torch.no_grad():
            self._binary_filter_pruning_mask.set_(mask)

    def forward(self, weight, update_weight=True):
        # In case of None weight (or bias) mask shouldn't be applied
        if weight is None:
            return weight

        # For weights self.mask_applying_dim should be used, for bias dim=0
        dim = 0 if not update_weight else self.mask_applying_dim

        if is_tracing_state():
            with no_jit_trace():
                return inplace_apply_filter_binary_mask(self.binary_filter_pruning_mask, weight, dim=dim)
        new_weight = apply_filter_binary_mask(self.binary_filter_pruning_mask, weight, dim=dim)
        return new_weight


def broadcast_filter_mask(filter_mask, shape, dim=0):
    broadcasted_shape = np.ones(len(shape), dtype=np.int64)
    broadcasted_shape[dim] = filter_mask.size(0)
    broadcasted_filter_mask = torch.reshape(filter_mask, tuple(broadcasted_shape))
    return broadcasted_filter_mask


def inplace_apply_filter_binary_mask(filter_mask: torch.Tensor,
                                     conv_weight: torch.nn.Parameter,
                                     node_name_for_logging: NNCFNodeName, dim=0):
    """
    Inplace applying binary filter mask to weight (or bias) of the convolution
    (by first dim of the conv weight).
    :param filter_mask: binary mask (should have the same shape as first dim of conv weight)
    :param conv_weight: weight or bias of convolution
    :return: result with applied mask
    """
    if filter_mask.size(0) != conv_weight.size(dim):
        raise RuntimeError("Shape of mask = {} for module {} isn't broadcastable to weight shape={}."
                           " ".format(filter_mask.shape, node_name_for_logging, conv_weight.shape))
    broadcasted_filter_mask = broadcast_filter_mask(filter_mask, conv_weight.shape, dim)
    return conv_weight.mul_(broadcasted_filter_mask)


def apply_filter_binary_mask(filter_mask, conv_weight, module_name="", dim=0):
    """
    Applying binary filter mask to weight (or bias) of the convolution (applying by first dim of the conv weight)
    without changing the weight.
    :param filter_mask: binary mask (should have the same shape as first dim of conv weight)
    :param conv_weight: weight or bias of convolution
    :return: result with applied mask
    """
    if filter_mask.size(0) != conv_weight.size(dim):
        raise RuntimeError("Shape of mask = {} for module {} isn't broadcastable to weight shape={}."
                           " ".format(filter_mask.shape, module_name, conv_weight.shape))

    broadcasted_filter_mask = broadcast_filter_mask(filter_mask, conv_weight.shape, dim)
    return broadcasted_filter_mask * conv_weight
