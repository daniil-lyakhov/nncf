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


from collections import defaultdict
from typing import List, Tuple, Union

import torch
import torch.fx
from torch.ao.quantization.pt2e.prepare import _get_edge_or_node_to_group_id
from torch.ao.quantization.pt2e.prepare import _get_edge_or_node_to_qspec
from torch.ao.quantization.quantizer import Quantizer as TorchAOQuantizer
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import QuantizationPointBase
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.quantization.quantizers.quantizer import Quantizer
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter

EdgeOrNode = Union[Tuple[torch.fx.Node, torch.fx.Node]]


class TorchAOQuantizerAdapter(Quantizer):
    """
    Implementation of the NNCF Quantizer interface for any given torch.ao quantizer.
    """

    def __init__(self, quantizer: TorchAOQuantizer):
        self._quantizer = quantizer

    def transform_prior_quantization(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        return self._quantizer.transform_for_annotation(model)

    def get_quantization_setup(self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        # Save model and nodes meta before the annotation
        original_meta = model.meta.copy()
        node_name_vs_meta = {}
        with torch.no_grad():
            for node in model.graph.nodes:
                node_name_vs_meta[node.name] = node.meta.copy()

        annotated_model = self._quantizer.annotate(model)
        self._quantizer.validate(annotated_model)
        quantizer_setup = self.get_quantizer_config_from_annotated_model(annotated_model)

        # Recover original meta
        model.meta = original_meta
        for node in model.graph.nodes:
            node.meta = node_name_vs_meta[node.name]

        return quantizer_setup

    @staticmethod
    def _get_quantization_points(
        from_node: torch.fx.Node,
        to_nodes: List[torch.fx.Node],
        anotated_model: torch.fx.GraphModule,
        qconfig: QuantizerConfig,
    ) -> List[QuantizationPointBase]:
        to_n = to_nodes[0]
        if from_node.op == "get_attr":
            _, metatype = GraphConverter.get_node_type_and_metatype(to_n, anotated_model)
            # Check that the constant is placed on the actual weight port, as it is possible for
            # activations to be a constant as well.
            if TorchAOQuantizerAdapter._get_node_args(to_n).index(from_node) in metatype.weight_port_ids:
                qip = WeightQuantizationInsertionPoint(to_n.name)
                return [SingleConfigQuantizationPoint(qip, qconfig, [x.name for x in to_nodes])]

        if len(from_node.users) == len(to_nodes):
            qip = ActivationQuantizationInsertionPoint(from_node.name)
            return [SingleConfigQuantizationPoint(qip, qconfig, [x.name for x in to_nodes])]

        qps = []
        for to_n_ in to_nodes:
            input_port_id = to_n_.args.index(from_node)
            qip = ActivationQuantizationInsertionPoint(to_n_.name, input_port_id)
            qp = SingleConfigQuantizationPoint(qip, qconfig, [to_n_.name])
            qps.append(qp)
        return qps

    @staticmethod
    def _get_node_args(node: torch.fx.Node):
        if node.target == torch.ops.aten.cat.default:
            return node.args[0]
        return node.args

    @staticmethod
    def get_quantizer_config_from_annotated_model(anotated_model: torch.fx.GraphModule) -> SingleConfigQuantizerSetup:
        edge_or_node_to_qspec = _get_edge_or_node_to_qspec(anotated_model)
        # Node means all output edges should be quantized.
        # Edge means only one edge should be quantized.
        edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)

        group_id_vs_edges = defaultdict(set)
        group_id_vs_qspec = {}
        for edge_or_node, group_id in edge_or_node_to_group_id.items():
            target_edges = [edge_or_node]
            if isinstance(edge_or_node, torch.fx.Node):
                target_edges = []
                for user in edge_or_node.users:
                    target_edges.append((edge_or_node, user))
            group_id_vs_edges[group_id].update(target_edges)
            # All qspecs should be aligned after the _get_edge_or_node_to_group_id call
            group_id_vs_qspec[group_id] = _unwrap_shared_qspec(
                edge_or_node_to_qspec[edge_or_node], edge_or_node_to_qspec
            )

        q_setup = SingleConfigQuantizerSetup()
        for group_id, edges in group_id_vs_edges.items():
            qspec = group_id_vs_qspec[group_id]
            if qspec is None:
                continue
            if not isinstance(qspec, QuantizationSpec):
                raise nncf.InternalError(f"Unknown torch.ao quantization spec: {qspec}")

            if qspec.qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                per_channel = True
            elif qspec.qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                per_channel = False
            else:
                raise nncf.InternalError(f"Unknown qscheme: {qspec.qscheme}")

            signed = qspec.dtype is torch.int8
            mode = (
                QuantizationMode.SYMMETRIC
                if qspec.qscheme in [torch.per_channel_symmetric, torch.per_tensor_symmetric]
                else QuantizationMode.ASYMMETRIC
            )
            narrow_range = qspec.quant_min % 2 != 0
            qconfig = QuantizerConfig(
                mode=mode, signedness_to_force=signed, per_channel=per_channel, narrow_range=narrow_range
            )

            joined_edges = defaultdict(list)
            for edge in edges:
                joined_edges[edge[0]].append(edge[1])

            qps = []
            for from_node, to_nodes in joined_edges.items():
                qps.extend(
                    TorchAOQuantizerAdapter._get_quantization_points(from_node, to_nodes, anotated_model, qconfig)
                )
            qp_ids = []
            for qp in qps:
                qp_ids.append(q_setup.add_independent_quantization_point(qp))
            if len(qp_ids) > 1:
                q_setup.register_unified_scale_group(qp_ids)

        return q_setup


def _unwrap_shared_qspec(qspec, edge_or_node_to_qspec):
    MAX_DEPTH = 1000
    i = 0
    while i < MAX_DEPTH and isinstance(qspec, SharedQuantizationSpec):
        qspec = edge_or_node_to_qspec[qspec.edge_or_node]
        i += 1
    if i == MAX_DEPTH:
        raise RuntimeError(f"Shared qspecs referenced to each other more than the limit: {MAX_DEPTH}")
    return qspec
