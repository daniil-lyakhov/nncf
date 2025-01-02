# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

import nncf
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.quantize_functions import TuneRange


def strip_tuned_lora_model(model: NNCFNetwork) -> NNCFNetwork:
    layout = model.nncf.transformation_layout()
    model = model.nncf.get_clean_shallow_copy()
    graph = model.nncf.get_graph()
    transformation_layout = TransformationLayout()
    t = layout.transformations
    # breakpoint()
    test_sorting = False
    if test_sorting:
        idx = [
            (idx, a.target_points[0].target_node_name)
            for idx, a in enumerate(t)
            if "q_proj" in a.target_points[0].target_node_name
        ][0][0]
        buff = t[0]
        t[0] = t[idx]
        t[idx] = buff

    for command in t:
        quantizer_module = command.fn
        if isinstance(quantizer_module, AsymmetricQuantizer):
            input_range_safe = abs(quantizer_module.input_range) + quantizer_module.eps
            input_low, input_range = TuneRange.apply(
                quantizer_module.input_low, input_range_safe, quantizer_module.levels
            )
            assert len(command.target_points) == 1
            tp = command.target_points[0]

            node_with_weight = graph.get_node_by_name(tp.target_node_name)

            weight_node = get_const_node(node_with_weight, tp.input_port_id, graph)
            weight_name = weight_node.layer_attributes.name
            module_name, weight_attr_name = split_const_name(weight_name)
            module = get_module_by_name(module_name, model)
            w = getattr(module, weight_attr_name)
            if w is None or not isinstance(w, torch.nn.Parameter):
                raise nncf.InternalError(f"Could not find a torch.nn.Parameter in the model by name {weight_name}.")

            input_ = w + quantizer_module._lora_B @ quantizer_module._lora_A
            input_ = input_.reshape(quantizer_module._group_shape)  # NOTE: careful with what you reshape here!

            scale = (quantizer_module.levels - 1) / input_range
            output = input_.clip(min=input_low, max=input_low + input_range)
            output -= input_low
            output *= scale

            # breakpoint()
            zero_point = (-input_low * scale).round()
            output_dtype = output.dtype
            output -= zero_point
            output = output.round()
            output = output.to(torch.int8) + zero_point.to(torch.int8)
            output = output.to(output_dtype)

            original_shape = w.shape
            compressor_scale = 1 / scale

            decompressor = INT4AsymmetricWeightsDecompressor(
                scale=compressor_scale,
                zero_point=zero_point.to(torch.uint8),
                compressed_weight_shape=output.shape,
                result_shape=original_shape,
                result_dtype=w.dtype,
            )

            packed_tensor = decompressor.pack_weight(output.to(torch.uint8))

            # tmp = decompressor(packed_tensor)

            # sets compressed tensor
            compressed_parameter = torch.nn.Parameter(packed_tensor, requires_grad=False)
            setattr(module, weight_attr_name, compressed_parameter)

            consumer_nodes = graph.get_next_nodes(weight_node)
            if len(consumer_nodes) > 1:
                for c_node in consumer_nodes:
                    c_module = model.get_module_by_scope(Scope.from_str(c_node.layer_name))
                    for name, param in c_module.named_parameters(recurse=False, remove_duplicate=False):
                        if id(param) == id(w):
                            setattr(c_module, name, compressed_parameter)

            # registry weight decompression module in the model
            decompressor_name = f"weights_decompressor_{weight_node.node_name.replace('.', '_')}"

            # inserts the weight decompressor into the model as the post hook on the model weight
            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    [PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=weight_node.node_name)],
                    decompressor,
                    decompressor_name,
                )
            )

    return PTModelTransformer(model).transform(transformation_layout)
