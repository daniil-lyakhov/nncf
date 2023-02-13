"""
 Copyright (c) 2023 Intel Corporation
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

from nncf.common.utils.registry import Registry
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import PatternNames
from nncf.torch.graph.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.torch.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import GROUP_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.torch.graph.pattern_operations import RELU_OPERATIONS
from nncf.torch.graph.patterns import create_fc_conv_mul
from nncf.torch.graph.patterns import create_h_sigmoid_act
from nncf.torch.graph.patterns import create_h_swish_act
from nncf.torch.graph.patterns import create_swish_act
from nncf.torch.graph.patterns import create_l2_norm


PT_HW_FUSED_PATTERNS = Registry('torch')

# ATOMIC OPERATIONS


@PT_HW_FUSED_PATTERNS.register(PatternNames.L2_NORM)
def create_l2_norm_operations():
    return create_l2_norm()

# COMBINATIONS


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic_operations():
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@PT_HW_FUSED_PATTERNS.register(PatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations_operations():
    batch_norm = batch_norm_operations()
    activations = activation_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@PT_HW_FUSED_PATTERNS.register(PatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm_operations():
    batch_norm = batch_norm_operations()
    activations = activation_operations()
    activations.join_patterns(batch_norm)
    return activations


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BATCH_NORM)
def create_linear_batch_norm_operations():
    linear = linear_operations()
    batch_norm = batch_norm_operations()
    linear.join_patterns(batch_norm)
    return linear


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS)
def create_linear_activation_operations():
    linear = linear_operations()
    activation = activation_operations()
    linear.join_patterns(activation)
    return linear


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BATCH_NORM_ACTIVATIONS)
def create_linear_batch_norm_activation_operations():
    linear_bn = create_linear_batch_norm_operations()
    activations = activation_operations()
    linear_bn.join_patterns(activations)
    return linear_bn


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS_BATCH_NORM)
def create_linear_activation_batch_norm_activations():
    linear_act = create_linear_activation_operations()
    batch_norm = batch_norm_operations()
    linear_act.join_patterns(batch_norm)
    return linear_act


@PT_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_BATCH_NORM)
def create_arithmetic_batch_norm_operations():
    arithmetic = arithmetic_operations()
    batch_norm = batch_norm_operations()
    arithmetic.join_patterns(batch_norm)
    return arithmetic


@PT_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations_operations():
    arithmetic = arithmetic_operations()
    activation = activation_operations()
    arithmetic.join_patterns(activation)
    return arithmetic


@PT_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS)
def create_arithmetic_batch_norm_activations_operations():
    arithmetic_bn = create_arithmetic_batch_norm_operations()
    activation = activation_operations()
    arithmetic_bn.join_patterns(activation)
    return arithmetic_bn


@PT_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_ACTIVATIONS_BATCH_NORM)
def create_arithmetic_activations_batch_norm_operations():
    arithmetic_act = create_arithmetic_activations_operations()
    batch_norm = batch_norm_operations()
    arithmetic_act.join_patterns(batch_norm)
    return arithmetic_act


@PT_HW_FUSED_PATTERNS.register(PatternNames.GROUP_NORM_RELU)
def create_group_norm_relu_operations():
    group_norm = GraphPattern()
    group_norm.add_node(**GROUP_NORMALIZATION_OPERATIONS)
    relu = GraphPattern()
    relu.add_node(**RELU_OPERATIONS)
    group_norm.join_patterns(relu)
    return group_norm


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_CONST_MULTIPLY)
def create_linear_const_multiply():
    return create_fc_conv_mul()


def linear_operations():
    pattern = GraphPattern()
    pattern.add_node(**LINEAR_OPERATIONS)
    return pattern

def arithmetic_operations():
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern


def batch_norm_operations():
    pattern = GraphPattern()
    pattern.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    return pattern


def activation_operations():
    atomic_activations = GraphPattern()
    atomic_activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    swish = create_swish_act()
    h_sigmoid = create_h_sigmoid_act()
    h_swish = create_h_swish_act()

    pattern = GraphPattern()
    pattern.add_pattern_alternative(atomic_activations)
    pattern.add_pattern_alternative(swish)
    pattern.add_pattern_alternative(h_swish)
    pattern.add_pattern_alternative(h_sigmoid)
    return pattern
