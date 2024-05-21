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
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.logging import nncf_logger
from nncf.experimental.torch_fx.transformations import separate_conv_and_bias
from nncf.experimental.torch_fx.transformations import separate_linear_and_bias
from nncf.experimental.torch_fx.transformations import view_to_reshape
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES


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
                node_type = str(node.target.overloadpacket).split(".")[1]
            elif node.target.__name__ == "getitem":
                node_type = "__getitem__"
            else:
                # TODO: get correct nodes types from this nodes as well
                node_type = str(node.target)
            node_metatype = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
        else:
            node_type = node.op
            node_metatype = UnknownMetatype
        if node_metatype is UnknownMetatype:
            nncf_logger.info(f"Unknown metatype for node: {node}")
        return node_type, node_metatype

    @staticmethod
    def create_nncf_graph(model: torch.fx.GraphModule) -> NNCFGraph:
        """
        Creates NNCFGraph from GraphModule.
        All nodes from model which have valid metatype are added to NNCFGraph.
        Then, corresponding edges are added to the NNCFGraph with shape, type, output and input port ids.

        :param model: torch fx GraphModule.
        :return: NNCFGraph.
        """

        _fuse_conv_bn_(model)
        # BN fuses to conv bias, conv+bias joined op
        # needs to be splited for nncf
        separate_linear_and_bias(model)
        separate_conv_and_bias(model)
        view_to_reshape(model)

        nncf_graph = PTNNCFGraph()

        for source_node in model.graph.nodes:

            node_type, node_metatype = GraphConverter._get_node_type_and_metatype(source_node)

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
            for idx, dist_node in enumerate(source_node.users):
                dist_node_id = nncf_graph.get_node_by_name(dist_node.name).node_id
                input_port_id, output_port_id, tensor_shape = GraphConverter.get_edge_params(
                    model, source_node, source_nncf_node, dist_node, idx
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
    def get_edge_params(
        model, source_node: torch.fx.Node, source_nncf_node: NNCFNode, dist_node: torch.fx.Node, output_idx: int
    ):
        output_port_id = 0
        if source_node.op in ("get_attr",):
            tensor_shape = tuple(getattr(model, source_node.target).shape)
        elif "val" in source_node.meta:
            if source_nncf_node.metatype is om.PTBatchNormMetatype:
                tensor = source_node.meta["val"][0]
            elif source_nncf_node.metatype is om.PTSplitMetatype:
                tensor = source_node.meta["val"][output_idx]
                # Assume every split outputs corresponds to an unique output_port_id
                output_port_id = output_idx
            else:
                tensor = source_node.meta["val"]
            tensor_shape = tuple(tensor.shape)
        else:
            nncf_logger.info(
                f"Edge shape between {source_node.name} and {dist_node.name} is unknown. Using [1,1,1,1] instead."
            )
            tensor_shape = [1, 1, 1, 1]

        input_port_id = dist_node.all_input_nodes.index(source_node)
        return input_port_id, output_port_id, tensor_shape
