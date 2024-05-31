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

from itertools import chain
from typing import Tuple

import torch.fx
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_  # noqa
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from torch.ao.quantization.pt2e.utils import _is_conv  # noqa

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES

# from nncf.experimental.torch_fx.operator_metatypes import FX_OPERATOR_METATYPES


class GraphConverter:
    """
    Builds the NNCFGraph from an OpenVINO model.
    """

    @staticmethod
    def _get_leaf_node(module: torch.nn.Module, node: torch.fx.Node) -> torch.nn.Module:
        py_obj = module
        assert isinstance(node.target, str)
        atoms = node.target.split(".")
        for atom in atoms:
            if not hasattr(py_obj, atom):
                raise RuntimeError(str(py_obj) + " does not have attribute " + atom + "!")
            py_obj = getattr(py_obj, atom)
        return py_obj

    @staticmethod
    def _get_node_type_and_metatype(node: torch.fx.Node) -> Tuple[str, om.OperatorMetatype]:
        if node.op == "placeholder":
            node_type = "input"
            node_metatype = om.PTInputNoopMetatype
        elif node.op == "output":
            node_type = "output"
            node_metatype = om.PTOutputNoopMetatype
        elif node.op == "get_attr":
            node_type = "get_attr"
            node_metatype = om.PTConstNoopMetatype
        elif node.op in ("call_function",):
            if hasattr(node.target, "overloadpacket"):
                torch.nn.BatchNorm2d
                node_type = str(node.target.overloadpacket).split(".")[1]
            elif node.target.__name__ == "getitem":
                node_type = "__getitem__"
            else:
                # TODO: get correct nodes types from this nodes as well
                node_type = str(node.target)
            node_metatype = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
            # TODO: add layer attrs and support subtypes
            # if node_metatype.get_subtypes():
            #    subtype = node_metatype.determine_subtype(
            #        #dynamic_graph_node.layer_attributes, functions_kwargs=dynamic_graph_node.__dict__
            #    )
        else:
            node_type = node.op
            node_metatype = UnknownMetatype
        return node_type, node_metatype

    @staticmethod
    def separate_conv_and_bias(model: torch.fx.GraphModule):
        """
        Separates one joined conv+bias node to two nodes: conv and bias.
        Needed as nncf does not expect joined conv
        """
        add_node_target = torch.ops.aten.add_.Tensor
        for n in model.graph.nodes:
            if not _is_conv(n):
                continue
            if len(n.args) < 3 or n.args[2] is None:
                continue
            conv_node = n
            dims = len(_get_tensor_constant_from_node(conv_node.args[1], model).shape)
            conv_bias_node = conv_node.args[2]
            conv_bias_value = _get_tensor_constant_from_node(conv_bias_node, model)
            args = list(n.args)
            args[2] = None
            conv_node.args = tuple(args)
            with model.graph.inserting_after(conv_node):
                new_conv_bias_node = create_getattr_from_value(
                    model,
                    model.graph,
                    conv_bias_node.name + "_",
                    conv_bias_value.reshape(
                        (
                            1,
                            -1,
                        )
                        + (1,) * (dims - 2)
                    ),
                )
            with model.graph.inserting_after(new_conv_bias_node):
                add_node = model.graph.create_node(
                    "call_function", add_node_target, (conv_node, new_conv_bias_node), {}
                )
            for user in list(conv_node.users):
                if user is add_node:
                    continue
                user.replace_input_with(conv_node, add_node)

            if "val" in conv_node.meta:
                add_node.meta["val"] = conv_node.meta["val"]
        model.graph.eliminate_dead_code()
        model.recompile()

    @staticmethod
    def merge_conv_and_bias(model: torch.fx.GraphModule):
        """
        Separates one joined conv+bias node to two nodes: conv and bias.
        Needed as nncf does not expect joined conv
        """
        add_node_targets = (torch.ops.aten.add_.Tensor,)
        for n in model.graph.nodes:
            if not _is_conv(n):
                continue
            if len(n.args) > 2 and n.args[2] is not None:
                continue
            bias_node = next(iter(n.users))
            if len(n.users) > 1 or bias_node.target not in add_node_targets:
                continue
            conv_node = n
            const_node = None
            for node in bias_node.all_input_nodes:
                if node is not conv_node:
                    const_node = node
                    break
            assert const_node is not None
            bias_value = _get_tensor_constant_from_node(const_node, model).squeeze()
            with model.graph.inserting_before(conv_node):
                new_bias_node = create_getattr_from_value(model, model.graph, const_node.name + "_", bias_value)
            args = list(conv_node.args)
            args[2] = new_bias_node
            conv_node.args = tuple(args)
            for user in list(bias_node.users):
                user.replace_input_with(bias_node, conv_node)

        model.graph.eliminate_dead_code()
        model.recompile()

    @staticmethod
    def create_nncf_graph(model: torch.fx.GraphModule) -> NNCFGraph:
        """
        Creates NNCFGraph from GraphModule.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.

        :param model: OpenVINO model.
        :return: NNCFGraph.
        """

        _fuse_conv_bn_(model)
        # BN fuses to conv bias, conv+bias joined op
        # needs to be splited for nncf
        GraphConverter.separate_conv_and_bias(model)

        nncf_graph = PTNNCFGraph()

        for source_node in model.graph.nodes:

            print(source_node.name, source_node.op, source_node.target, sep=" ")
            node_type, node_metatype = GraphConverter._get_node_type_and_metatype(source_node)
            print(node_metatype)

            nncf_node = nncf_graph.add_nncf_node(
                node_name=source_node.name,
                node_type=node_type,
                node_metatype=node_metatype,  # layer_attributes,
            )

            def get_module_params_or_buffers():
                for pname, ptensor in chain(leaf_module.named_parameters(), leaf_module.named_buffers()):
                    pname1 = source_node.name + "." + pname
                    nncf_param_node = nncf_graph.add_nncf_node(
                        pname1,
                        "parameter" if isinstance(ptensor, torch.nn.Parameter) else "buffer",
                        om.PTConstNoopMetatype,
                    )
                    # TODO: Use valid tensor_shape, input_port_id, output_port_id
                    nncf_graph.add_edge_between_nncf_nodes(
                        nncf_param_node, nncf_node, tensor_shape=[1, 1, 1, 1], input_port_id=0, output_port_id=0
                    )

            if source_node.op == "call_module":
                leaf_module = GraphConverter._get_leaf_node(model, source_node)

                if not isinstance(leaf_module, torch.fx.GraphModule):
                    get_module_params_or_buffers()

        for source_node in model.graph.nodes:

            source_nncf_node = nncf_graph.get_node_by_name(source_node.name)
            for dist_node in source_node.users:
                dist_node_id = nncf_graph.get_node_by_name(dist_node.name).node_id
                input_port_id, output_port_id, tensor_shape = GraphConverter.get_edge_params(
                    model, source_node, source_nncf_node, dist_node
                )

                nncf_graph.add_edge_between_nncf_nodes(
                    source_nncf_node.node_id,
                    dist_node_id,
                    tensor_shape=tensor_shape,
                    input_port_id=input_port_id,
                    output_port_id=output_port_id,
                    dtype=Dtype.FLOAT,
                )

        return nncf_graph

    @staticmethod
    def get_edge_params(model, source_node: torch.fx.Node, source_nncf_node: NNCFNode, dist_node: torch.fx.Node):
        # TODO: support cat
        output_port_id = 0
        if source_node.op in ("get_attr",):
            tensor_shape = tuple(getattr(model, source_node.target).shape)
        elif "val" in source_node.meta:
            if source_nncf_node.metatype is om.PTBatchNormMetatype:
                tensor = source_node.meta["val"][0]
            else:
                tensor = source_node.meta["val"]
            tensor_shape = tuple(tensor.shape)
        else:
            print(f"Edge shape between {source_node.name} and {dist_node.name} is unknown. Using [1,1,1,1] instead.")
            tensor_shape = [1, 1, 1, 1]

        input_port_id = dist_node.all_input_nodes.index(source_node)
        return input_port_id, output_port_id, tensor_shape
