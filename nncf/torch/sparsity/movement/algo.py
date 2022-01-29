"""
 Copyright (c) 2019-2020 Intel Corporation
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
from copy import deepcopy
from typing import List

import torch
import torch.distributed as dist

from nncf import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.torch.algo_selector import PT_COMPRESSION_ALGORITHMS
from nncf.api.compression import CompressionStage
from nncf.common.graph import NNCFNode
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import BaseSparsityAlgoBuilder, BaseSparsityAlgoController, SparseModuleInfo
from nncf.torch.sparsity.movement.layers import MovementSparsifyingWeight, SparseConfig, SparseStructure
from nncf.torch.sparsity.movement.loss import ImportanceLoss, SparseLossForPerLayerSparsity
from nncf.torch.utils import get_world_size
from nncf.common.utils.helpers import matches_any
from nncf.common.accuracy_aware_training.training_loop import ADAPTIVE_COMPRESSION_CONTROLLERS
from nncf.torch.sparsity.collector import PTSparseModelStatisticsCollector
from nncf.common.sparsity.schedulers import SPARSITY_SCHEDULERS
from nncf.common.schedulers import StubCompressionScheduler
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.torch.search_building_blocks.search_blocks import get_building_blocks
from collections import namedtuple
from nncf.torch.dynamic_graph.operation_address import OperationAddress
import networkx as nx
from nncf.torch.layers import NNCF_MODULES_OP_NAMES
import os


@PT_COMPRESSION_ALGORITHMS.register('movement_sparsity')
class MovementSparsityBuilder(BaseSparsityAlgoBuilder):
    def create_weight_sparsifying_operation(self, target_module_node: NNCFNode, compression_lr_multiplier: float):
        sparse_cfg=None
        if 'sparse_structure_by_scopes' in self._algo_config:
            for sparse_mode, sparse_args, regex in self._algo_config['sparse_structure_by_scopes']:
                if matches_any(target_module_node.node_name, regex):
                    sparse_cfg = SparseConfig(sparse_mode, sparse_args)
                    break

        if sparse_cfg is None:
            sparse_cfg = SparseConfig()

        return MovementSparsifyingWeight(
                    target_module_node.layer_attributes.get_weight_shape(), 
                    frozen=False,
                    compression_lr_multiplier=compression_lr_multiplier,
                    eps=1e-6, 
                    sparse_cfg=sparse_cfg)

    def _build_controller(self, model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return MovementSparsityController(model, self._sparsified_module_info, self.config)


@ADAPTIVE_COMPRESSION_CONTROLLERS.register('pt_movement_sparsity')
class MovementSparsityController(BaseSparsityAlgoController):
    def __init__(self, target_model: NNCFNetwork, sparsified_module_info: List[SparseModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model, sparsified_module_info)
        algo_config = extract_algo_specific_config(config, 'movement_sparsity')
        params = deepcopy(algo_config.get('params', {}))

        self._distributed = False
        self._mode = params.get('sparsity_level_setting_mode', 'global')
        self._check_sparsity_masks = params.get('check_sparsity_masks', False)

        sparsify_operations = [m.operand for m in self.sparsified_module_info]
        if self._mode == 'local':
            # TODO: make sure we test this loop out
            self._loss = SparseLossForPerLayerSparsity(sparsify_operations)
            self._scheduler = StubCompressionScheduler()
        else:
            scheduler_cls = SPARSITY_SCHEDULERS.get(params.get('schedule', 'exponential')) #TODO: can we actually map to other scheduler in current implementation
            self._scheduler = scheduler_cls(self, params)
            self._loss = ImportanceLoss(sparsify_operations, self.scheduler)

        #TODO: review - perhaps not the right place
        self.config = config
        self.prunableops_per_group = self._get_group_of_prunable_ops()
        self.visualize_groups_of_prunables()

    def compression_stage(self) -> CompressionStage:
        if self._mode == 'local':
            return CompressionStage.FULLY_COMPRESSED

        if self.scheduler.current_sparsity_level == 0:
            return CompressionStage.UNCOMPRESSED
        if self.scheduler.current_sparsity_level >= self.scheduler.target_level:
            return CompressionStage.FULLY_COMPRESSED
        return CompressionStage.PARTIALLY_COMPRESSED

    def freeze(self):
        self._loss.disable()

    def distributed(self):
        if not dist.is_initialized():
            raise KeyError('Could not set distributed mode for the compression algorithm '
                           'because the default process group has not been initialized.')

        if next(self._model.parameters()).is_cuda:
            state = torch.cuda.get_rng_state()
            if dist.get_backend() == dist.Backend.NCCL:
                state = state.cuda()
            torch.distributed.broadcast(state, src=0)
            torch.cuda.set_rng_state(state.cpu())
        else:
            state = torch.get_rng_state()
            torch.distributed.broadcast(state, src=0)
            torch.set_rng_state(state)

        self._distributed = True

    def _check_distributed_masks(self):
        if not self._distributed or get_world_size() == 1:
            return 1

        nvalues = 0
        ncor_values = 0
        eps = 1e-4
        for minfo in self.sparsified_module_info:
            mask = minfo.operand.mask

            mask_list = [torch.empty_like(mask) for _ in range(get_world_size())]
            # nccl does not support gather, send, recv operations
            dist.all_gather(mask_list, mask)

            for i in range(1, len(mask_list)):
                rel_error = (mask_list[0] - mask_list[i]) / mask_list[0]
                ncor_values = ncor_values + (rel_error.abs() < eps).sum(dtype=mask.dtype)
                nvalues = nvalues + mask_list[i].numel()

        return ncor_values / nvalues

    def statistics(self, quickly_collected_only=False) -> NNCFStatistics:
        collector = PTSparseModelStatisticsCollector(self.model, self.sparsified_module_info)
        model_statistics = collector.collect()

        stats = MovementSparsityStatistics(model_statistics,
                                           self.scheduler.current_importance_threshold,
                                           self.scheduler.current_importance_lambda)

        nncf_stats = NNCFStatistics()
        nncf_stats.register('movement_sparsity', stats)
        return nncf_stats

    @property
    def compression_rate(self):
        return self.statistics().movement_sparsity.model_statistics.sparsity_level

    def _propagate_masks(self):
        # nncf_logger.debug("MVMT - Propagating pruning masks")
        # 1. Propagate masks for all modules
        from collections import OrderedDict
        sparse_sd = OrderedDict()
        with torch.no_grad():    
            for sparse_info in self.sparsified_module_info:
                for n, m in self.model.named_modules():
                    if m == sparse_info.module:
                        # print(n, 1-sparse_info.operand.binary_mask.count_nonzero()/sparse_info.operand.binary_mask.numel())
                        # print("pre", 1-m.weight.count_nonzero()/m.weight.numel())
                        # print("mask", 1-sparse_info.operand.binary_mask.count_nonzero()/sparse_info.operand.binary_mask.numel())
                        sparse_sd[n+'.weight'] = m.weight*sparse_info.operand.binary_mask
                        # print("post", 1-sparse_sd[n+'.weight'].count_nonzero()/sparse_sd[n+'.weight'].numel())
                # sd = sparse_info.module.state_dict()
                # sd['weight'] = sparse_info.module.weight*sparse_info.operand.binary_mask
                # sparse_info.module.load_state_dict(sd)

        model_sd = self.model.state_dict()
        for k, v in sparse_sd.items():
            assert k in model_sd, "key not exists!"
            model_sd[k] = sparse_sd[k]
        self.model.load_state_dict(model_sd)

        # init_output_masks_in_graph(graph, self.pruned_module_groups_info.get_all_nodes())
        # MaskPropagationAlgorithm(graph, PT_PRUNING_OPERATOR_METATYPES).mask_propagation()

        # # 2. Set the masks for Batch/Group Norms
        # pruned_node_modules = []
        # for node, pruning_block, node_module in self._pruned_norms_operators:
        #     if node_module not in pruned_node_modules:
        #         # Setting masks for BN nodes
        #         pruning_block.binary_filter_pruning_mask = node.data['output_mask'].tensor
        #         pruned_node_modules.append(node_module)

    def prepare_for_export(self):
        """
        Applies pruning masks to layer weights before exporting the model to ONNX.
        """
        self._propagate_masks()


    def print_prunableops_per_group(self):
        for group, op_list in self.prunableops_per_group.items():
            print("= Group {} ======".format(group))
            print('\n'.join(list(map(lambda x: '{:12} | {}'.format(str(list(x.op_mod.weight.shape)), str(x.op_addr)), op_list))))
  
    def _get_group_of_prunable_ops(self):
        PrunableOp = namedtuple("PrunableOp", "op_addr op_mod")

        building_blocks  = get_building_blocks(self.model, allow_nested_blocks=False)
        all_node_op_addr_in_blocks = self._get_all_node_op_addresses_in_block(self.model, building_blocks)

        prunableops_per_group = {}
        for group_id, nodes_per_block in all_node_op_addr_in_blocks.items():
            prunableops_per_group[group_id] = []

            for str_op_addr in nodes_per_block:
                op_address = OperationAddress.from_str(str_op_addr)
                if op_address.operator_name in NNCF_MODULES_OP_NAMES:

                    prunableops_per_group[group_id].append(
                        PrunableOp(
                            op_address,
                            self.model.get_module_by_scope(op_address.scope_in_model)
                        )
                    )
        return prunableops_per_group

    def _get_all_node_op_addresses_in_block(self, nncf_network, blocks):
        graph = nncf_network.get_original_graph()
        all_nodes_per_skipped_block_idxs = {}
        for idx, block in enumerate(blocks):
            start_node, end_node = block
            start_node_key, end_node_key = None, None
            for node in graph._nx_graph._node.values():
                if start_node == str(node['node_name']):
                    start_node_key = node['key']
                if end_node == str(node['node_name']):
                    end_node_key = node['key']
            simple_paths = nx.all_simple_paths(graph._nx_graph, start_node_key, end_node_key)
            all_nodes_in_block = set()
            for node_keys_in_path in simple_paths:
                for node_key in node_keys_in_path:
                    all_nodes_in_block.add(str(graph._nx_graph._node[node_key]['node_name']))
            start_op_address = str(graph._nx_graph._node[start_node_key]['node_name'])
            all_nodes_in_block.remove(start_op_address)
            all_nodes_per_skipped_block_idxs[idx] = list(all_nodes_in_block)
        return all_nodes_per_skipped_block_idxs

    def visualize_groups_of_prunables(self, path=None):
        import networkx as nx
        from nncf.torch.graph.graph import PTNNCFGraph
        from networkx.drawing.nx_agraph import to_agraph
        import matplotlib._color_data as mcd
        import matplotlib.pyplot as plt
        import numpy as np
        palette = np.array(list(mcd.CSS4_COLORS.keys())).reshape(-1, 4).transpose().reshape(-1).tolist()

        from matplotlib.colors import to_hex
        palette = np.array([to_hex(c) for c in plt.get_cmap("tab20b").colors]).reshape(-1, 5).transpose().reshape(-1).tolist()
        
        learnable_node_color_map = dict()
        opbook = dict()

        for group_id, op_list in self.prunableops_per_group.items():
            color = palette[group_id % len(palette)]
            for op in op_list:
                learnable_node_color_map[str(op.op_addr)] = color
                opbook[str(op.op_addr)] = op

        building_blocks  = get_building_blocks(self.model, allow_nested_blocks=False)
        node_op_address_per_block = self._get_all_node_op_addresses_in_block(self.model, building_blocks)
        node_color_map = dict()
        for group_id, op_list in node_op_address_per_block.items():
            color = palette[group_id % len(palette)]
            for op in op_list:
                node_color_map[op] = color

        g = self.model.get_graph()

        out_graph = nx.DiGraph()
        for node_name, node in g._nx_graph.nodes.items():
            # ia_op_exec_context = node[PTNNCFGraph.IA_OP_EXEC_CONTEXT_NODE_ATTR]

            attrs_node = {}
            label = node['key']
            # label = str(node[PTNNCFGraph.ID_NODE_ATTR]) + ' ' + str(ia_op_exec_context)
            # if 'conv2d' in label.lower():
            #     label = "*prunable*\n" + label
            tokens=label.split("/")
            new_tokens=[]
            for i, token in enumerate(tokens):
                if (i+1)%2==0:
                    token += "\n"
                new_tokens.append(token)
            attrs_node['label'] = '/'.join(new_tokens)

            if node['node_name'] in node_color_map:
                # cluster_id = self.df.cluster_id[self.df.node_name == node_name].values[0]
                # attrs_node['label'] += "\n(cluster {})".format(cluster_id)
                # mcd.CSS4_COLORS
                # attrs_node['color'] = mcd.CSS4_COLORS[node_color_map[node['node_name']]]

                
                attrs_node['color'] = node_color_map[node['node_name']]
                if node['node_name'] in learnable_node_color_map:
                    attrs_node['label'] += "\n{}\n".format(str(tuple(opbook[node['node_name']].op_mod.weight.shape)))
                    attrs_node['style'] = 'filled'
                else:
                    attrs_node['style'] = 'diagonals'
                    # At present, there are 8 style values recognized: filled , invisible , diagonals , rounded . dashed , dotted , solid and bold

            out_graph.add_node(node_name, **attrs_node)

        for u, v in g._nx_graph.edges:
            out_graph.add_edge(u, v, label=g._nx_graph.edges[u, v][PTNNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        if path is None:
            path = 'mvmt_prunableops_group_viz.dot'
        path = os.path.join(self.config.get("log_dir", "."), path)
        
        nx.drawing.nx_pydot.write_dot(out_graph, path)

        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            print("Graphviz is not installed - only the .dot model visualization format will be used. "
                                "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                                "PNG rendering.")