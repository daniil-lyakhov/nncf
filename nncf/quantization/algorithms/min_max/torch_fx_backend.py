# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.quantization.fake_quantize import FakeQuantize

import nncf
import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.quantizers import calculate_asymmetric_level_ranges
from nncf.common.quantization.quantizers import calculate_symmetric_level_ranges
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.tensor_statistics.collectors import REDUCERS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.model_utils import get_target_point
from nncf.experimental.torch.fx.transformations import qdq_insertion_transformation_builder
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeConvertParameters
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import ELEMENTWISE_OPERATIONS
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import TuneRange
from nncf.torch.quantization.layers import get_scale_shape
from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high
from nncf.torch.quantization.strip import convert_to_torch_fakequantizer
from nncf.torch.utils import no_jit_trace


class FXMinMaxAlgoBackend(MinMaxAlgoBackend):
    @property
    def preserved_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def mat_mul_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTLinearMetatype, om.PTMatMulMetatype]

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def dropout_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTDropoutMetatype]

    @property
    def read_variable_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def conv_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTConv1dMetatype, om.PTConv2dMetatype, om.PTConv3dMetatype]

    @property
    def elementwise_metatypes(self) -> List[OperatorMetatype]:
        return ELEMENTWISE_OPERATIONS

    @property
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.PTConv1dMetatype,
            om.PTConv2dMetatype,
            om.PTConv3dMetatype,
            om.PTLinearMetatype,
            om.PTConvTranspose1dMetatype,
            om.PTConvTranspose2dMetatype,
            om.PTConvTranspose3dMetatype,
        ]

    @property
    def add_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTAddMetatype]

    @property
    def group_conv_metatypes(self) -> List[OperatorMetatype]:
        return self.conv_metatypes

    @property
    def scaled_dot_product_attention_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTScaledDotProductAttentionMetatype]

    @property
    def scales_unification_map(self) -> Dict[OperatorMetatype, OperatorMetatype]:
        return {om.PTCatMetatype: self.overflow_fix_metatypes + self.scaled_dot_product_attention_metatypes}

    @property
    def hw_config(self) -> HWConfig:
        return PTHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT

    @property
    def reducer_map(self) -> Dict[StatisticsType, TensorReducerBase]:
        return REDUCERS_MAP

    @property
    def supports_inplace_statistics(self) -> bool:
        return False

    @staticmethod
    def get_start_nodes_for_activation_path_tracing(nncf_graph: PTNNCFGraph) -> List[NNCFNode]:
        return nncf_graph.get_input_nodes()

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return get_target_point(target_type, target_node_name, port_id)

    @staticmethod
    def create_convert_insertion_command(
        target_point: PTTargetPoint,
        parameters: FakeConvertParameters,
    ) -> TransformationCommand:
        raise nncf.InternalError("FakeConvert insertion not implemented in PyTorch backend!")

    @staticmethod
    def get_target_point_shape(nncf_graph: PTNNCFGraph, node: NNCFNode, target_point: PTTargetPoint) -> Tuple[int, ...]:
        return nncf_graph.get_input_shape_for_insertion_point(target_point)

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: PTTargetPoint, ndims: int) -> Tuple[int]:
        # TODO(dlyakhov): support transpose conv and other cases
        return (0,)

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Optional[int]]:
        return get_weight_tensor_port_ids(node, graph)

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: PTTargetPoint) -> str:
        weighted_node = nncf_graph.get_node_by_name(target_point.target_node_name)
        weight_edge = nncf_graph.get_input_edge_by_port_id(weighted_node, target_point.input_port_id)
        weight = weight_edge.from_node
        return weight.node_name

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: NNCFNetwork) -> QuantizerConfig:
        return config

    @staticmethod
    def _get_input_scale_shape(
        nncf_graph: NNCFGraph, target_point: PTTargetPoint, per_channel: bool
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        is_weights = target_point.is_weight_target_point()
        if is_weights:
            # TODO(dlyakhov): support transpose conv/ make channel_idx common
            channel_idx = 0
        else:
            channel_idx = 1  # channel dim for activations

        input_shape = nncf_graph.get_input_shape_for_insertion_point(target_point)
        scale_shape = tuple(
            get_scale_shape(input_shape, is_weights=is_weights, per_channel=per_channel, channel_idx=channel_idx)
        )

        return input_shape, scale_shape, channel_idx

    @staticmethod
    def _create_native_torch_quantizers(
        quantizer_config: QuantizerConfig,
        scale_shape: Tuple,
        parameters: FakeQuantizeParameters,
        target_type: TargetType,
    ):
        ch_axis = int(torch.argmax(torch.tensor(scale_shape)))
        eps = 1e-16
        half_range = False

        if quantizer_config.mode == QuantizationMode.SYMMETRIC:
            signed = bool(torch.any(parameters.input_low.data < 0))
        else:
            signed = True

        # Overwrite signed in case of signedness_to_force is specified
        if quantizer_config.signedness_to_force is not None:
            signed = quantizer_config.signedness_to_force

        scaled_num_bits = int(half_range)
        if quantizer_config.mode == QuantizationMode.SYMMETRIC:

            level_low, level_high = calculate_symmetric_level_ranges(
                quantizer_config.num_bits - scaled_num_bits, signed, quantizer_config.narrow_range
            )
            qscheme = torch.per_channel_symmetric if quantizer_config.per_channel else torch.per_tensor_symmetric

            scale = torch.nn.Parameter((parameters.input_high.data - eps).reshape(scale_shape))
            quant_min, quant_max, scale, zero_point = FXMinMaxAlgoBackend._get_parameters_for_torch_fq_symmetric(
                level_low, level_high, scale, eps
            )
        elif quantizer_config.mode == QuantizationMode.ASYMMETRIC:
            if not signed:
                level_low, level_high = calculate_asymmetric_level_ranges(quantizer_config.num_bits - scaled_num_bits)
            else:
                level_low = -127 if quantizer_config.narrow_range else -128
                level_high = 127
            qscheme = torch.per_channel_affine if quantizer_config.per_channel else torch.per_tensor_affine
            quant_min, quant_max, scale, zero_point = FXMinMaxAlgoBackend._get_parameters_for_torch_fq_asymmetric(
                level_low,
                level_high,
                parameters.levels,
                parameters.input_low.data,
                parameters.input_high.data,
                eps,
            )

        dtype = torch.qint8 if level_low < 0 else torch.quint8
        if quantizer_config.per_channel:
            observer = torch.ao.quantization.observer.PerChannelMinMaxObserver
        else:
            observer = torch.ao.quantization.observer.MinMaxObserver

        fakequantizer = FakeQuantize(
            observer=observer,
            quant_max=quant_max,
            quant_min=quant_min,
            dtype=dtype,
            qscheme=qscheme,
            eps=eps,
        )

        if not quantizer_config.per_channel:
            scale = scale.squeeze()
            zero_point = zero_point.squeeze()

        fakequantizer.scale = scale
        fakequantizer.ch_axis = ch_axis
        fakequantizer.zero_point = zero_point

        # Disable observer to save parameters
        fakequantizer.disable_observer()

        return fakequantizer

    @staticmethod
    def _get_parameters_for_torch_fq_asymmetric(
        level_low, level_high, levels, input_low, input_high, eps
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """
        Get parameters for conversion to native FakeQuantize.

        :return: A Tuple
            quant_max - Fixed the low quant number.
            quant_min - Fixed the high quant number.
            scale - Quantizer scale.
            zero_point - Quantizer zero point.
        """

        with torch.no_grad(), no_jit_trace():

            input_range = input_high - input_low
            input_range_safe = torch.abs(input_range) + eps
            input_low, input_range_tuned = TuneRange.apply(input_low, input_range_safe, levels)
            input_high = input_low + input_range_tuned

            scale, zero_point = get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high)

            scale = scale.view(-1)
            zero_point = zero_point.view(-1).to(dtype=torch.int32)

        return level_low, level_high, scale, zero_point

    @staticmethod
    def _get_parameters_for_torch_fq_symmetric(
        level_low, level_high, scale, eps
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """
        Get parameters for conversion to native FakeQuantize.

        :return: A Tuple
            quant_max - Fixed the low quant number.
            quant_min - Fixed the high quant number.
            scale - Quantizer scale.
            zero_point - Quantizer zero point.
        """

        with torch.no_grad(), no_jit_trace():
            input_range = abs(scale) + eps
            input_low = input_range * level_low / level_high
            input_high = input_range

            scale, zero_point = get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high)

            scale = scale.view(-1)
            zero_point = zero_point.view(-1).to(dtype=torch.int32)

        return level_low, level_high, scale, zero_point

    @staticmethod
    def _create_quantizer(
        quantizer_config: QuantizerConfig,
        scale_shape: Tuple,
        parameters: FakeQuantizeParameters,
        target_type: TargetType,
    ) -> FakeQuantize:
        mode = quantizer_config.mode
        quantizer_cls = QUANTIZATION_MODULES.get(mode)
        quantizer_spec = PTQuantizerSpec.from_config(
            quantizer_config,
            narrow_range=quantizer_config.narrow_range,
            scale_shape=scale_shape,
            half_range=False,
            logarithm_scale=False,
            is_quantized_on_export=False,
            compression_lr_multiplier=None,
        )
        quantizer = quantizer_cls(quantizer_spec)

        # Fill it with minmax
        # TODO(dlyakhov) Prevent creation of intermediate objects like nncf quantizer.
        FXMinMaxAlgoBackend._fill_quantizer_parameters(quantizer, parameters, quantizer_spec.scale_shape)
        # Convert to the torch fake quantizer
        torch_fq = convert_to_torch_fakequantizer(quantizer)
        return torch_fq

    @staticmethod
    def _fill_quantizer_parameters(quantizer: BaseQuantizer, parameters: FakeQuantizeParameters, scale_shape) -> None:
        if isinstance(quantizer, AsymmetricQuantizer):
            quantizer.input_low = torch.nn.Parameter(parameters.input_low.data.reshape(scale_shape))
            input_range = parameters.input_high - parameters.input_low
            # Subtract eps from the input_range to make quantizer parameters equal to
            # original parameters on the forward call.
            quantizer.input_range = torch.nn.Parameter((input_range.data - quantizer.eps).reshape(scale_shape))
        else:
            if quantizer._signedness_to_force is None:
                quantizer.signed = bool(torch.any(parameters.input_low.data < 0))
            else:
                quantizer.signed = quantizer._signedness_to_force
            # Subtract eps from the scale to make quantizer parameters equal to
            # original parameters on the forward call.
            quantizer.scale = torch.nn.Parameter((parameters.input_high.data - quantizer.eps).reshape(scale_shape))

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> FXApplyTransformationCommand:
        _, scale_shape, _ = FXMinMaxAlgoBackend._get_input_scale_shape(
            nncf_graph, target_point, quantizer_config.per_channel
        )

        quantizer = FXMinMaxAlgoBackend._create_native_torch_quantizers(
            quantizer_config, scale_shape, parameters, target_point.target_type
        )
        # quantizer = FXMinMaxAlgoBackend._create_quantizer(
        #    quantizer_config, scale_shape, parameters, target_point.target_type
        # )
        transformation = qdq_insertion_transformation_builder(quantizer, [target_point])
        return FXApplyTransformationCommand(transformation)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: List[PTTargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> List[PTSharedFnInsertionCommand]:
        _, scale_shape, _ = FXMinMaxAlgoBackend._get_input_scale_shape(
            nncf_graph, target_points[0], quantizer_config.per_channel
        )

        # quantizer = FXMinMaxAlgoBackend._create_quantizer(
        #    quantizer_config, scale_shape, parameters, target_points[0].target_type
        # )
        quantizer = FXMinMaxAlgoBackend._create_native_torch_quantizers(
            quantizer_config, scale_shape, parameters, target_points[0].target_type
        )

        transformations = []
        for tp in target_points:
            transformation = qdq_insertion_transformation_builder(quantizer, [tp])
            transformations.append(FXApplyTransformationCommand(transformation))
        return transformations

    @staticmethod
    def get_ignored_metatypes(model_type: ModelType, device: TargetDevice) -> List[OperatorMetatype]:
        types = []
        if model_type == ModelType.TRANSFORMER:
            types = [
                om.PTAddMetatype,
                om.PTPowerMetatype,
                om.PTSubMetatype,
                om.PTAvgPool2dMetatype,
                om.PTAvgPool3dMetatype,
                om.PTMeanMetatype,
                om.PTSumMetatype,
                om.PTReduceL2,
                om.PTDivMetatype,
                om.PTMaxMetatype,
                om.PTSqueezeMetatype,
                om.PTLayerNormMetatype,
                om.PTModuleLayerNormMetatype,
                om.PTGroupNormMetatype,
                om.PTModuleGroupNormMetatype,
                # Batchnorm
                om.PTBatchNormMetatype,
                om.PTModuleBatchNormMetatype,
            ]
            if device != TargetDevice.CPU_SPR:
                types.append(om.PTMulMetatype)
        return types

    @staticmethod
    def get_ignored_names_by_layer_attributes(nncf_graph: NNCFGraph) -> Set[str]:
        return set()

    def get_weight_nodes(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        weight_nodes_candidates = [
            node
            for node in nncf_graph.get_all_nodes()
            if issubclass(node.metatype, om.PTOperatorMetatype) and node.metatype.weight_port_ids
        ]
        weight_nodes = []
        for node in weight_nodes_candidates:
            if node.metatype in self.mat_mul_metatypes and not self.is_matmul_with_constant(node, nncf_graph):
                continue
            weight_nodes.append(node)
        return weight_nodes

    def is_matmul_with_constant(self, node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return node.metatype in self.mat_mul_metatypes and len(get_weight_tensor_port_ids(node, nncf_graph)) > 0
