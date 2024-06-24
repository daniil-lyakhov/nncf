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

from typing import Callable, List, Optional

import torch
import torch.fx
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.quantization.fake_quantize import FakeQuantize

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.torch_fx.model_transformer import FXModelTransformer
from nncf.torch.graph.transformations.commands import PTTargetPoint


def stat_collectorts_insertion_tranformation_builder():
    def stat_collectorts_insertion_tranformation(model: torch.fx.GraphModule, node: torch.fx.Node):
        pass

    return stat_collectorts_insertion_tranformation


def fake_quantize_insertion_tranformation_builder(quantizer: FakeQuantize, target_points: List[PTTargetPoint]):
    def fake_quantize_insertion_transformation(model: torch.fx.GraphModule):
        module_attr_name = _set_module_to_the_graph_module(model, quantizer, target_points)
        graph = model.graph
        for target_point in target_points:
            target_node = FXModelTransformer._get_target_node(model.graph, target_point)
            with graph.inserting_after(target_node):
                fq_node = graph.create_node(
                    "call_module", module_attr_name, (target_node,), {}, name=module_attr_name + "_quantizer"
                )
            for user in list(target_node.users):
                if user is fq_node:
                    continue
                user.replace_input_with(target_node, fq_node)

    return fake_quantize_insertion_transformation


def bias_update_transformation_builder(node: NNCFNode, value: torch.Tensor):
    def bias_update_transformation(model: torch.fx.GraphModule):
        graph = model.graph
        target_node_name = node.node_name
        graph_node = FXModelTransformer.get_graph_node_by_name(graph, target_node_name)
        bias_node = next(iter(graph_node.users))
        with graph.inserting_before(bias_node):
            new_constant = create_getattr_from_value(model, graph, target_node_name + "_shifted_bias", value)
        args = list(bias_node.args)
        args[1] = new_constant
        bias_node.args = tuple(args)
        graph.eliminate_dead_code()

    return bias_update_transformation


def qdq_insertion_tranformation_builder(quantizer: FakeQuantize, target_points: List[PTTargetPoint]):
    def qdq_insertion_tranformation(model: torch.fx.GraphModule):
        if any(tp.target_type != TargetType.OPERATION_WITH_WEIGHTS for tp in target_points) and len(target_points) > 1:
            raise RuntimeError
        for target_point in target_points:
            target_node = FXModelTransformer._get_target_node(model.graph, target_point)
            insert_one_qdq(model, target_node, quantizer, target_point)

    return qdq_insertion_tranformation


def insert_one_qdq(
    model: torch.fx.GraphModule, target_node: torch.fx.Node, quantizer: FakeQuantize, target_point: PTTargetPoint
):
    # 1. extract information for inserting q/dq node from activation_post_process
    node_type = "call_function"
    quantize_op: Optional[Callable] = None
    # scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
    dtype = torch.int8 if quantizer.quant_min < 0 else torch.uint8
    if quantizer.is_per_channel:
        qparams = {
            "_scale_": quantizer.scale,
            "_zero_point_": quantizer.zero_point,
            "_axis_": quantizer.ch_axis,
            "_quant_min_": quantizer.quant_min,
            "_quant_max_": quantizer.quant_max,
            "_dtype_": dtype,
        }
        quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
        dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
    else:
        qparams = {
            "_scale_": float(quantizer.scale),
            "_zero_point_": int(quantizer.zero_point),
            "_quant_min_": quantizer.quant_min,
            "_quant_max_": quantizer.quant_max,
            "_dtype_": dtype,
        }
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    # Quantized functions accepts only uint8 as an input
    # if target_point.target_type != TargetType.OPERATION_WITH_WEIGHTS and qparams["_dtype_"] == torch.int8:
    #    breakpoint()
    #    raise RuntimeError("Wrong parameters: activations should always be uint8")

    # TODO: map FakeQuantizePramaeters to qparams for quantize/dequantize
    # 2. replace activation_post_process node with quantize and dequantize
    graph = model.graph
    # TODO: use metatype to get correct input_port_id
    # Do not quantize already quantized nodes
    # inserting_before handle only order in the graph generated code.
    # so, inserting quantize-dequantize and all constant nodes before the usage of the nodes
    with graph.inserting_before(target_node):
        quantize_op_inputs = [target_node]
        for key, value_or_node in qparams.items():
            # TODO: we can add the information of whether a value needs to
            # be registered as an attribute in qparams dict itself
            if key in ["_scale_", "_zero_point_"] and (not isinstance(value_or_node, (float, int))):
                # For scale and zero_point values we register them as buffers in the root module.
                # However, note that when the values are not tensors, as in the case of
                # per_tensor quantization, they will be treated as literals.
                # However, registering them as a node seems to cause issue with dynamo
                # tracing where it may consider tensor overload as opposed to default.
                # With extra check of scale and zero_point being scalar, it makes
                # sure that the default overload can be used.
                # TODO: maybe need more complex attr name here
                qparam_node = create_getattr_from_value(model, graph, target_node.name + key, value_or_node)
                quantize_op_inputs.append(qparam_node)
            else:
                # for qparams that are not scale/zero_point (like axis, dtype) we store
                # them as literals in the graph.
                quantize_op_inputs.append(value_or_node)
    with graph.inserting_after(target_node):
        quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
        # use the same qparams from quantize op
    dq_inputs = [quantized_node] + quantize_op_inputs[1:]
    user_dq_nodes = []
    with graph.inserting_after(quantized_node):
        for user in target_node.users:
            if user is quantized_node:
                continue
            user_dq_nodes.append((user, graph.call_function(dequantize_op, tuple(dq_inputs), {})))

    for user, dq_node in user_dq_nodes:
        user.replace_input_with(target_node, dq_node)


def _set_module_to_the_graph_module(
    model: torch.fx.GraphModule, module_to_insert: torch.nn.Module, target_points: List[PTTargetPoint]
) -> str:
    """
    Sets given module to the given torch.fx.GraphModule with unique name.
    """
    module_to_insert = module_to_insert
    module_name_in_model = (
        ";".join(
            "_".join((tp.target_node_name, str(tp.input_port_id), str(tp.target_type.value))) for tp in target_points
        )
        + "_"
        + str(id(module_to_insert))
    )
    assert not hasattr(model, module_name_in_model)
    setattr(model, module_name_in_model, module_to_insert)
    return module_name_in_model
