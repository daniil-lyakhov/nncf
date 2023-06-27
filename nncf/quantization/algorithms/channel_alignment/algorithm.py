# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from tqdm import tqdm

from nncf import Dataset
from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.channel_alignment.backend import ConvParamsContainer
from nncf.quantization.algorithms.channel_alignment.backend import DimsDescriptor
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")

FAST_BIAS_CORRECTION_THRESHOLD = 2


class ChannelAlignment(Algorithm):
    """
    Post-training FastBiasCorrection algorithm implementation.

    The main purpose of this algorithm to reduce quantization error
    via correction the bias of the Convolutions, FullyConnected, etc. layers.
    The algorithm pipeline is very simple:
        - we collects floating-point statistics from the corresponding model for the layers with bias;
        - then we gets the quantized model and try to reduce it's error by correction of the bias;
        - the shift calculates using the sub-graph that consists of the correction layer and
        weight quantizer-dequantizer pair or fake quantize node;
        - the floating-point statistics uses as input for
        the sub-graph and further quantization output calculation;
        - in the end we corrects the original bias by the difference (shift)
        between floating-point and quantized outputs.
    """

    def __init__(
        self,
        subset_size: int = 100,
        inplace_statistics: bool = True,
        backend_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param subset_size: Size of a subset for the statistics collection,
            defaults to 100.
        :param threshold: The magnitude threshold that regulates the application of the
            shift. Magnitude calculates as the maximum of the absolute ratio of the
            shift to the original bias value. If the calculated value is less than the
            threshold, the shift will apply to the bias, defaults to 2.
        :param apply_for_all_nodes: If True, then the bias correction be applied to all
            quantized nodes, if the node has no bias then a bias node will be inserted,
            and if False, then the bias correction will only be applied to quantized
            nodes that have a bias.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param backend_params: Backend specific parameters.
        """
        super().__init__()
        self.subset_size = subset_size
        self.inplace_statistics = inplace_statistics
        self.backend_params = backend_params
        self.nncf_graph = None
        self._backend_entity = None
        self._nncf_grpah = None
        self._q = 1e-4

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.channel_alignment.openvino_backend import OVChannelAlignmentAlgoBackend

            self._backend_entity = OVChannelAlignmentAlgoBackend()

    def _apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)

        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        model_transformer = ModelTransformerFactory.create(model)
        transformation_layout = TransformationLayout()

        def filter_func(point: StatisticPoint) -> bool:
            return ChannelAlignment in point.algorithm_to_tensor_collectors and point.target_point == target_point

        for conv_in, add_in, conv_out in tqdm(self._get_node_pairs(nncf_graph), desc="Channel allignment"):
            target_point, node_in = self._get_target_point_and_node_in(conv_in, add_in)
            tensor_collectors = list(
                statistic_points.get_algo_statistics_for_node(node_in.node_name, filter_func, ChannelAlignment)
            )
            assert len(tensor_collectors) == 1
            stat: MinMaxTensorStatistic = tensor_collectors[0].get_statistics()

            conv_in_cont = ConvParamsContainer(conv_in, model, nncf_graph, self._backend_entity)
            conv_out_cont = ConvParamsContainer(conv_out, model, nncf_graph, self._backend_entity)
            dims_descriptor: DimsDescriptor = self._backend_entity.get_dims_descriptor(conv_in)
            if conv_in_cont.has_bias() and conv_out_cont.has_bias():
                amean = (stat.max_values + stat.min_values) * 0.5
                conv_in_cont.bias, conv_out_cont.bias = self._align_means(
                    conv_in_cont.bias, conv_out_cont.bias, conv_out_cont.weight, amean, dims_descriptor
                )

            ascale = (stat.max_values - stat.min_values).astype(np.float32)
            eps = np.finfo(ascale.dtype).eps
            if (ascale > eps).any():
                conv_in_cont.weight, conv_out_cont.weight, conv_in_cont.bias = self._align_scales(
                    conv_in_cont.weight,
                    conv_out_cont.weight,
                    conv_in_cont.bias,
                    ascale,
                    dims_descriptor,
                    eps,
                )

            for container in [conv_in_cont, conv_out_cont]:
                if not np.equal(container.weight, container.original_weight).all():
                    transformation_layout.register(
                        self._backend_entity.create_weights_update_command(
                            container.op, container.weight, container.weight_port_id
                        )
                    )

                if not np.equal(container.bias, container.original_bias).all():
                    transformation_layout.register(
                        self._backend_entity.create_bias_update_command(container.op, container.bias, nncf_graph)
                    )

        transformed_model = model_transformer.transform(transformation_layout)
        return transformed_model

    @staticmethod
    def _align_means(
        bias_in_value: np.ndarray,
        bias_out_value: np.ndarray,
        conv_out_value: np.ndarray,
        amean: np.ndarray,
        dims_descriptor: DimsDescriptor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        updated_add_in_value = bias_in_value - amean.reshape(bias_in_value.shape)

        weight_dims = conv_out_value.ndim
        updated_conv_out_value = conv_out_value
        if weight_dims > 2:
            axes = list(range(weight_dims))
            axes.remove(dims_descriptor.conv_weight_in_channels_dim)
            axes.remove(dims_descriptor.conv_weight_out_channels_dim)
            updated_conv_out_value = np.sum(conv_out_value, axis=tuple(axes))

        out_channel_dim, in_channel_dim = 0, 1
        if dims_descriptor.conv_weight_out_channels_dim > dims_descriptor.conv_weight_in_channels_dim:
            out_channel_dim, in_channel_dim = in_channel_dim, out_channel_dim

        updated_conv_out_value = np.transpose(
            updated_conv_out_value,
            (out_channel_dim, in_channel_dim),
        )
        shift = updated_conv_out_value.dot(amean.reshape(updated_conv_out_value.shape[1]))

        updated_add_out_value = bias_out_value + shift.reshape(bias_out_value.shape)
        return updated_add_in_value, updated_add_out_value

    @staticmethod
    def _align_scales(
        conv_in_value: np.ndarray,
        conv_out_value: np.ndarray,
        bias_in_value: np.ndarray,
        ascale: np.ndarray,
        dims_descr: DimsDescriptor,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # scale producer convolution weights
        conv_in_shape = conv_in_value.shape
        if conv_in_shape[dims_descr.conv_weight_out_channels_dim] == ascale.shape[dims_descr.bias_channels_dim]:
            positive_scales_mask = ascale > eps
            scale_factor = ascale / np.median(ascale[positive_scales_mask])
            scale_factor[~positive_scales_mask] = 1
            scale_factor = np.clip(scale_factor, 1e-2, 1e2)

            scale_in_shape = np.ones(len(conv_in_shape), dtype=int)
            scale_in_shape[dims_descr.conv_weight_out_channels_dim] = scale_factor.shape[dims_descr.bias_channels_dim]
            conv_in_value = conv_in_value / scale_factor.reshape(scale_in_shape)

            if bias_in_value is not None:
                bias_in_value = bias_in_value / scale_factor.reshape(bias_in_value.shape)

            scale_out_shape = np.ones(len(conv_out_value.shape), dtype=int)
            scale_out_shape[dims_descr.conv_weight_in_channels_dim] = scale_factor.shape[dims_descr.bias_channels_dim]
            conv_out_value = conv_out_value * scale_factor.reshape(scale_out_shape)
        return conv_in_value, conv_out_value, bias_in_value

    def _check_consumer_conv_node(self, conv_node: NNCFNode) -> bool:
        attrs = self._backend_entity.get_conv_layer_attributes(conv_node)
        if attrs is None:
            return False
        # Check groups amount == 1
        if attrs.groups != 1:
            return False
        # Check node has no padding
        if any(attrs.padding_values):
            return False
        # Check node has valid stride
        if any(elem != 1 for elem in attrs.stride):
            return False
        # Check Node has vaild dilation
        if any(elem != 1 for elem in attrs.dilations):
            return False
        return True

    def _check_producer_conv_node(self, conv_node: NNCFNode):
        attrs = self._backend_entity.get_conv_layer_attributes(conv_node)
        if attrs is None:
            return False
        # Check groups amount == 1
        if attrs.groups != 1:
            return False
        return True

    def _get_target_patterns(self) -> GraphPattern:
        producer_attrs = {
            GraphPattern.LABEL_ATTR: "CONV_PRODUCER",
            GraphPattern.NODE_TYPE_ATTR: [
                op for op in self._backend_entity.get_conv_metatypes() + self._backend_entity.get_linear_metatypes()
            ],
        }
        bias_attrs = {
            GraphPattern.LABEL_ATTR: "BIAS_PRODUCER",
            GraphPattern.NODE_TYPE_ATTR: [op for op in self._backend_entity.get_add_metatypes()],
        }
        bias_const_attrs = {
            GraphPattern.LABEL_ATTR: "BIAS_CONSTANT",
            GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE,
        }
        consumer_attrs = {
            GraphPattern.LABEL_ATTR: "CONV_CONSUMER",
            GraphPattern.NODE_TYPE_ATTR: [op for op in self._backend_entity.get_conv_metatypes()],
        }
        conv_const_attrs = {
            GraphPattern.LABEL_ATTR: "CONV_CONSTANT",
            GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE,
        }

        def get_conv_conv_pattern() -> GraphPattern:
            conv_conv = GraphPattern()
            producer_constant = conv_conv.add_node(**conv_const_attrs)
            consumer_constant = conv_conv.add_node(**conv_const_attrs)

            pattern_conv_producer = conv_conv.add_node(**producer_attrs)
            pattern_conv_consumer = conv_conv.add_node(**consumer_attrs)

            conv_conv.add_edge(producer_constant, pattern_conv_producer)
            conv_conv.add_edge(consumer_constant, pattern_conv_consumer)

            conv_conv.add_edge(pattern_conv_producer, pattern_conv_consumer)
            return conv_conv

        def get_conv_add_conv_pattern() -> GraphPattern:
            conv_bias_conv = GraphPattern()
            producer_constant = conv_bias_conv.add_node(**conv_const_attrs)
            bias_producer_const = conv_bias_conv.add_node(**bias_const_attrs)
            consumer_constant = conv_bias_conv.add_node(**conv_const_attrs)

            pattern_conv_producer = conv_bias_conv.add_node(**producer_attrs)
            pattern_bias_producer = conv_bias_conv.add_node(**bias_attrs)
            pattern_conv_consumer = conv_bias_conv.add_node(**consumer_attrs)

            conv_bias_conv.add_edge(producer_constant, pattern_conv_producer)
            conv_bias_conv.add_edge(consumer_constant, pattern_conv_consumer)
            conv_bias_conv.add_edge(bias_producer_const, pattern_bias_producer)

            conv_bias_conv.add_edge(pattern_conv_producer, pattern_bias_producer)
            conv_bias_conv.add_edge(pattern_bias_producer, pattern_conv_consumer)
            return conv_bias_conv

        pattern = get_conv_conv_pattern()
        pattern.add_pattern_alternative(get_conv_add_conv_pattern())
        return pattern

    def _get_node_pairs(self, nncf_graph: NNCFGraph) -> List[Tuple[NNCFNode, Optional[NNCFNode], NNCFNode]]:
        pairs = []
        patterns = self._get_target_patterns()
        for subgraph in nncf_graph.find_matching_subgraphs(patterns):
            if len(subgraph) == 2:
                add_in = None
                conv_in, conv_out = subgraph
            else:
                conv_in, add_in, conv_out = subgraph

            if not self._check_producer_conv_node(conv_in):
                continue

            if not self._check_consumer_conv_node(conv_out):
                continue

            pairs.append((conv_in, add_in, conv_out))
        return pairs

    def _get_target_point_and_node_in(self, conv_in, add_in) -> Tuple[TargetPoint, NNCFNode]:
        node_in = conv_in if add_in is None else add_in
        input_port_id, _ = self._backend_entity.get_activation_port_ids_for_node(node_in)
        return (
            self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION, node_in.node_name, input_port_id),
            node_in,
        )

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        self.nncf_graph = NNCFGraphFactory.create(model)

        statistic_container = StatisticPointsContainer()
        for conv_in, add_in, _ in self._get_node_pairs(self.nncf_graph):
            target_point, node_in = self._get_target_point_and_node_in(conv_in, add_in)
            channel_axis = conv_in.metatype.output_channel_axis
            reduction_shape = list(range(len(self.nncf_graph.get_output_edges(node_in)[0].tensor_shape)))
            reduction_shape.remove(channel_axis)

            statistic_collector = self._backend_entity.get_statistic_collector(
                tuple(reduction_shape), self._q, self.subset_size, self.inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=target_point,
                    tensor_collector=statistic_collector,
                    algorithm=ChannelAlignment,
                )
            )

        return statistic_container
