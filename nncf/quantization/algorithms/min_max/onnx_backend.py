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

from typing import Dict, List, Tuple, Optional
import numpy as np
import onnx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType

from nncf.onnx.hardware.config import ONNXHWConfig
from nncf.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.onnx.quantization.quantizer_parameters import calculate_activation_quantizer_parameters
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.nncf_graph_builder import ONNXWeightedNodesLayerAttributes
from nncf.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXNonMaxSuppressionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXTopKMetatype
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector

from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS


@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [ONNXTopKMetatype, ONNXNonMaxSuppressionMetatype]

    @property
    def hw_config(self) -> HWConfig:
        return ONNXHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_activation_quantizer_insertion_command(nncf_graph: NNCFGraph,
                                                      target_point: ONNXTargetPoint,
                                                      quantizer_config: QuantizerConfig,
                                                      statistics: MinMaxTensorStatistic) \
                                                      -> ONNXQuantizerInsertionCommand:
        return ONNXMinMaxAlgoBackend._create_quantizer_insertion_command(nncf_graph,
                                                                         target_point,
                                                                         quantizer_config,
                                                                         statistics)

    @staticmethod
    def create_weight_quantizer_insertion_command(nncf_graph: NNCFGraph,
                                                  target_point: ONNXTargetPoint,
                                                  quantizer_config: QuantizerConfig,
                                                  statistics: MinMaxTensorStatistic) -> ONNXQuantizerInsertionCommand:
        return ONNXMinMaxAlgoBackend._create_quantizer_insertion_command(nncf_graph,
                                                                         target_point,
                                                                         quantizer_config,
                                                                         statistics)

    @staticmethod
    def _create_quantizer_insertion_command(nncf_graph: NNCFGraph,
                                            target_point: ONNXTargetPoint,
                                            quantizer_config: QuantizerConfig,
                                            statistics: MinMaxTensorStatistic):
        axis = ONNXMinMaxAlgoBackend._get_axis(nncf_graph,
                                               target_point,
                                               quantizer_config)
        parameters = calculate_activation_quantizer_parameters(statistics, quantizer_config, axis)
        return ONNXQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def _get_axis(nncf_graph: NNCFGraph,
                  target_point: ONNXTargetPoint,
                  quantizer_config: QuantizerConfig) -> Optional[int]:
        if not quantizer_config.per_channel:
            return None
        if target_point.is_activation_target_point():
            return 1
        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        return node.metatype.weight_definitions.weight_channel_axis

    @staticmethod
    def _get_reduction_shape_and_use_abs_max(nncf_graph: NNCFGraph,
                                             target_point: ONNXTargetPoint,
                                             quantizer_config: QuantizerConfig):

        use_abs_max = quantizer_config.mode == QuantizationMode.SYMMETRIC
        if not quantizer_config.per_channel:
            return None, use_abs_max

        if target_point.is_activation_target_point():
            # TODO: support reduction shapes for 3D-5D conv cases
            return (0, 2, 3), use_abs_max

        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        assert isinstance(node.layer_attributes, ONNXWeightedNodesLayerAttributes)
        weight_shape = node.layer_attributes.weight_shape
        reduction_shape = list(range(len(weight_shape)))

        axis = ONNXMinMaxAlgoBackend._get_axis(nncf_graph, target_point,
                                               quantizer_config)
        reduction_shape.pop(axis)
        return tuple(reduction_shape), use_abs_max

    @staticmethod
    def minmax_statistic_collector(nncf_graph: NNCFGraph,
                                   target_point: ONNXTargetPoint,
                                   quantizer_config: QuantizerConfig,
                                   num_samples: int = None) -> ONNXMinMaxStatisticCollector:
        reduction_shape, use_abs_max =\
            ONNXMinMaxAlgoBackend._get_reduction_shape_and_use_abs_max(nncf_graph,
                                                                       target_point,
                                                                       quantizer_config)
        return ONNXMinMaxStatisticCollector(use_abs_max, reduction_shape, num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(nncf_graph: NNCFGraph,
                                        target_point: ONNXTargetPoint,
                                        quantizer_config: QuantizerConfig,
                                        use_per_sample_stats: bool,
                                        num_samples: int = None) -> ONNXMeanMinMaxStatisticCollector:
        reduction_shape, use_abs_max =\
            ONNXMinMaxAlgoBackend._get_reduction_shape_and_use_abs_max(nncf_graph,
                                                                       target_point,
                                                                       quantizer_config)
        return ONNXMeanMinMaxStatisticCollector(use_per_sample_stats,
                                                use_abs_max,
                                                reduction_shape,
                                                num_samples,
                                                window_size=None)

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        return node.metatype.weight_definitions.weight_port_id
