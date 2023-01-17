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

import pytest

from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.definitions import Granularity
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.graph.nncf_graph_builder import ONNXWeightedNodesLayerAttributes
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from tests.common.quantization.test_filter_constant_nodes import create_mock_graph
from tests.common.quantization.test_filter_constant_nodes import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import NodeWithType


# pylint: disable=protected-access

class NNCFGraphToTest:
    def __init__(self, conv_layer_attrs = None):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Output_1
        nodes = [NodeWithType('Input_1', InputNoopMetatype),
                 NodeWithType('Conv_1', ONNXConvolutionMetatype, conv_layer_attrs),
                 NodeWithType('Output_1', OutputNoopMetatype),
                 ]
        node_edges = [('Input_1', 'Conv_1'), ('Conv_1', 'Output_1')]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)

    def set_layers_attributes(self):
        self.nncf_graph

class NNCFGraphToTestDepthwiseConv:
    def __init__(self):
        #       Original graph
        #          Input_1
        #             |
        #        DepthwiseConv_1
        #             |
        #           Output_1
        nodes = [NodeWithType('Input_1', InputNoopMetatype),
                 NodeWithType('Conv_1', ONNXDepthwiseConvolutionMetatype),
                 NodeWithType('Output_1', OutputNoopMetatype),
                 ]
        node_edges = [('Input_1', 'Conv_1'), ('Conv_1', 'Output_1')]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)


@pytest.mark.parametrize('nncf_graph', [NNCFGraphToTest()])
def test_default_quantizer_config(nncf_graph):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters())
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph.nncf_graph)

    weight_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                            num_bits=8,
                                            signedness_to_force=True,
                                            per_channel=True)
    activation_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                                num_bits=8,
                                                signedness_to_force=None,
                                                per_channel=False)

    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_weight_quantization_point():
            assert quantization_point.qconfig == weight_default_config
        if quantization_point.is_activation_quantization_point():
            assert quantization_point.qconfig == activation_default_config


@pytest.mark.parametrize('weight_granularity', [Granularity.PERCHANNEL, Granularity.PERTENSOR])
@pytest.mark.parametrize('activation_granularity', [Granularity.PERTENSOR])
@pytest.mark.parametrize('preset', [QuantizationPreset.MIXED, QuantizationPreset.PERFORMANCE])
@pytest.mark.parametrize('weight_bits', [8])
@pytest.mark.parametrize('activation_bits', [8])
@pytest.mark.parametrize('signed_weights', [None])
@pytest.mark.parametrize('signed_activations', [None])
@pytest.mark.parametrize('nncf_graph', [NNCFGraphToTest()])
# TODO(kshpv): add signed_activations and signed_weights which should be independent from HW config.
def test_quantizer_config_from_ptq_params(weight_granularity, activation_granularity, preset, weight_bits,
                                          activation_bits, signed_weights, signed_activations, nncf_graph):
    algo = PostTrainingQuantization(
        PostTrainingQuantizationParameters(preset=preset,
                                           weight_bits=weight_bits,
                                           weight_granularity=weight_granularity,
                                           signed_weights=signed_weights,
                                           activation_bits=activation_bits,
                                           activation_granularity=activation_granularity,
                                           signed_activations=signed_activations
                                           ))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph.nncf_graph)
    q_g_to_quantization_mode = {}
    for q_g in QuantizerGroup:
        q_g_to_quantization_mode[q_g] = preset.get_params_configured_by_preset(q_g)['mode']
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_weight_quantization_point():
            assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.WEIGHTS]
            assert quantization_point.qconfig.per_channel == (weight_granularity == Granularity.PERCHANNEL)
            assert quantization_point.qconfig.num_bits == weight_bits
            if signed_weights is not None:
                assert quantization_point.qconfig.signedness_to_force == signed_weights
        if quantization_point.is_activation_quantization_point():
            assert quantization_point.qconfig.per_channel == (activation_granularity == Granularity.PERCHANNEL)
            assert quantization_point.qconfig.num_bits == activation_bits
            assert quantization_point.qconfig.mode == q_g_to_quantization_mode[QuantizerGroup.ACTIVATIONS]
            if signed_activations is not None:
                assert quantization_point.qconfig.signedness_to_force == signed_activations


@pytest.mark.parametrize('nncf_graph', [NNCFGraphToTestDepthwiseConv()])
def test_depthwise_conv_default_quantizer_config(nncf_graph):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters())
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph.nncf_graph)

    weight_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                            num_bits=8,
                                            signedness_to_force=True,
                                            per_channel=True)
    activation_default_config = QuantizerConfig(mode=QuantizationMode.SYMMETRIC,
                                                num_bits=8,
                                                signedness_to_force=None,
                                                per_channel=True)

    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_weight_quantization_point():
            assert quantization_point.qconfig == weight_default_config
        if quantization_point.is_activation_quantization_point():
            assert quantization_point.qconfig == activation_default_config


def _get_target_point(target_type: TargetType) -> ONNXTargetPoint:
    return ONNXTargetPoint(target_type, target_node_name='/Conv_1_0', port_id=0)

@pytest.mark.parametrize('nncf_graph', [NNCFGraphToTest(
    conv_layer_attrs=ONNXWeightedNodesLayerAttributes('dummy', 'dummy', (4, 4, 4, 4)))])
@pytest.mark.parametrize('target_point', [_get_target_point(TargetType.POST_LAYER_OPERATION),
                                          _get_target_point(TargetType.OPERATION_WITH_WEIGHTS)])
@pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX])
@pytest.mark.parametrize('q_config_mode', [QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC])
@pytest.mark.parametrize('q_config_per_channel', [True, False])
def test_get_stat_collector(nncf_graph, target_point, range_type,
                            q_config_mode, q_config_per_channel):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    q_config = QuantizerConfig(num_bits=8,
                               mode=q_config_mode,
                               per_channel=q_config_per_channel)
    tensor_collector = min_max_algo._get_stat_collector(nncf_graph.nncf_graph,
                                                        target_point, q_config)

    is_weight_tp = target_point.is_weight_target_point()
    # tensor_collector type check
    if is_weight_tp or q_config_per_channel:
        assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)
    else:
        if range_type == RangeType.MINMAX:
            assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)
        elif range_type == RangeType.MEAN_MINMAX:
            assert isinstance(tensor_collector, ONNXMeanMinMaxStatisticCollector)

    # reduction_shape check
    if q_config_per_channel:
        ref_reduction_shape = (1, 2, 3) if is_weight_tp else (0, 2, 3)
        assert tensor_collector._reduction_shape == ref_reduction_shape
    else:
        assert tensor_collector._reduction_shape is None

    # use_abs_max check
    if q_config_mode == QuantizationMode.SYMMETRIC:
        assert tensor_collector._use_abs_max
    elif q_config_mode == QuantizationMode.ASYMMETRIC:
        assert not tensor_collector._use_abs_max
