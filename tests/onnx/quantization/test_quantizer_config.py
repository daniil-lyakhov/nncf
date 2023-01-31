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

from nncf.common.graph.transformations.commands import TargetType
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.onnx.graph.nncf_graph_builder import ONNXWeightedNodesLayerAttributes
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint

from tests.post_training.test_quantizer_config import TemplateTestQuantizerConfig
from tests.post_training.models import NNCFGraphToTest
from tests.post_training.models import NNCFGraphToTestDepthwiseConv


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return ONNXMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return ONNXMeanMinMaxStatisticCollector

    def get_target_point(self, target_type: TargetType, target_node_name) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id=0)

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        conv_layer_attrs = ONNXWeightedNodesLayerAttributes('dummy', 'dummy', (4, 4, 4, 4))
        return NNCFGraphToTest(ONNXConvolutionMetatype, conv_layer_attrs)

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return NNCFGraphToTestDepthwiseConv(ONNXDepthwiseConvolutionMetatype)
