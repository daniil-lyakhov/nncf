"""
 Copyright (c) 2020 Intel Corporation
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
import torch
from functools import partial
from functools import update_wrapper
from typing import List, Dict

from nncf.torch.module_operations import UpdateWeightAndBiasPruning

from torch import nn
from texttable import Texttable

from nncf import NNCFConfig
from nncf.config.extractors import extract_algo_specific_config
from nncf.torch.algo_selector import ZeroCompressionLoss
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.pruning.operations import PT_PRUNING_OPERATOR_METATYPES
from nncf.torch.pruning.filter_pruning.layers import apply_filter_binary_mask
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.clusterization import Cluster
from nncf.torch.pruning.structs import PrunedModuleInfo


class BasePruningAlgoBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        params = self._algo_config.get('params', {})
        self._set_default_params_for_ranking_type(params)
        self._params = params

        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_batch_norms = params.get('prune_batch_norms', True)
        self.prune_downsample_convs = params.get('prune_downsample_convs', False)

        self._prunable_types = self.get_op_types_of_pruned_modules()

        from nncf.common.pruning.pruning_node_selector import PruningNodeSelector
        self.pruning_node_selector = PruningNodeSelector(PT_PRUNING_OPERATOR_METATYPES,
                                                         self._prunable_types,
                                                         self.get_types_of_grouping_ops(),
                                                         self.ignored_scopes,
                                                         self.target_scopes,
                                                         self.prune_first,
                                                         self.prune_last,
                                                         self.prune_downsample_convs)

        self.pruned_module_groups_info = []
        self._pruned_norms_operators = {}

    @staticmethod
    def _set_default_params_for_ranking_type(params: Dict) -> None:
        """
        Setting default parameter values of pruning algorithm depends on the ranking type:
        for learned_ranking `all_weights` must be True (in case of False was set by the user, an Exception will be
        raised), `prune_first_conv`, `prune_last_conv`, `prune_downsample_convs` are recommended to be True (this
        params will be set to True by default (and remain unchanged if the user sets some value).
        :param params: dict with parameters of the algorithm from config
        """
        learned_ranking = 'interlayer_ranking_type' in params and params['interlayer_ranking_type'] == 'learned_ranking'
        if not learned_ranking:
            return
        nncf_logger.info('For learning global ranking `prune_first_conv`, `prune_last_conv`, `prune_downsample_convs`, '
                         '`all_weights` are setting to True by default. It is not recommended to set this params'
                         ' to False.')
        params.setdefault('prune_first_conv', True)
        params.setdefault('prune_last_conv', True)
        params.setdefault('prune_downsample_convs', True)
        if params.get('all_weights') is False:
            raise Exception('In case of `interlayer_ranking_type`=`learned_ranking`, `all_weights` must be set to True,'
                            ' plese, change this in config settings.')
        params.setdefault('all_weights', True)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = self._prune_weights(target_model)
        for command in commands:
            layout.register(command)
        return layout

    def _prune_weights(self, target_model: NNCFNetwork):
        target_model_graph = target_model.get_original_graph()
        groups_of_nodes_to_prune = self.pruning_node_selector.create_pruning_groups(target_model_graph)

        device = next(target_model.parameters()).device
        insertion_commands = []
        self.pruned_module_groups_info = Clusterization[PrunedModuleInfo](lambda x: x.node_name)

        for i, group in enumerate(groups_of_nodes_to_prune.get_all_clusters()):
            group_minfos = []
            for node in group.elements:
                node_name = node.node_name
                module = target_model.get_containing_module(node_name)
                module_scope = target_model_graph.get_scope_by_node_name(node_name)
                # Check that we need to prune weights in this op
                assert self._is_pruned_module(module)

                nncf_logger.info("Adding Weight Pruner in scope: {}".format(node_name))
                pruning_block = self.create_weight_pruning_operation(module)
                # Hook for weights and bias
                hook = UpdateWeightAndBiasPruning(pruning_block).to(device)
                insertion_commands.append(
                    PTInsertionCommand(
                        PTTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                      target_node_name=node_name),
                        hook,
                        TransformationPriority.PRUNING_PRIORITY
                    )
                )
                group_minfos.append(PrunedModuleInfo(node_name=node_name,
                                                     module_scope=module_scope,
                                                     module=module,
                                                     operand=pruning_block,
                                                     node_id=node.node_id))

            cluster = Cluster[PrunedModuleInfo](i, group_minfos, [n.node_id for n in group.elements])
            self.pruned_module_groups_info.add_cluster(cluster)

        # Adding binary masks also for Batch/Group Norms to allow applying masks after propagation
        all_norm_layers = target_model_graph.get_nodes_by_types(['batch_norm', 'group_norm'])
        for node in all_norm_layers:
            node_name = node.node_name
            module = target_model.get_containing_module(node_name)

            pruning_block = self.create_weight_pruning_operation(module)
            # Hook for weights and bias
            hook = UpdateWeightAndBiasPruning(pruning_block).to(device)
            insertion_commands.append(
                PTInsertionCommand(
                    PTTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                  target_node_name=node_name),
                    hook,
                    TransformationPriority.PRUNING_PRIORITY
                )
            )
            self._pruned_norms_operators[node_name] = (pruning_block, module)
        return insertion_commands

    def create_weight_pruning_operation(self, module):
        raise NotImplementedError

    def _is_pruned_module(self, module: nn.Module):
        """
        Return whether this module should be pruned or not.
        """
        raise NotImplementedError

    def get_op_types_of_pruned_modules(self):
        """
        Returns list of operation types that should be pruned.
        """
        raise NotImplementedError

    def get_types_of_grouping_ops(self):
        raise NotImplementedError

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class BasePruningAlgoController(PTCompressionAlgorithmController):
    def __init__(self, target_model: NNCFNetwork,
                 prunable_types: List[str],
                 pruned_module_groups_info: Clusterization[PrunedModuleInfo],
                 config: NNCFConfig):
        super().__init__(target_model)
        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        self._prunable_types = prunable_types
        self.config = config
        self.pruning_config = extract_algo_specific_config(config, 'filter_pruning')
        params = self.pruning_config.get('params', {})
        self.pruned_module_groups_info = pruned_module_groups_info
        self.prune_batch_norms = params.get('prune_batch_norms', True)
        self.prune_first = params.get('prune_first_conv', False)
        self.prune_last = params.get('prune_last_conv', False)
        self.prune_downsample_convs = params.get('prune_downsample_convs', False)
        self.zero_grad = params.get('zero_grad', True)
        self.prune_flops = False
        self.check_pruning_rate(params)
        self._hooks = []

    def freeze(self):
        raise NotImplementedError

    def set_pruning_rate(self, pruning_rate):
        raise NotImplementedError

    def zero_grads_for_pruned_modules(self):
        """
        This function registers a hook that will set the
        gradients for pruned filters to zero.
        """
        self._clean_hooks()

        def hook(grad, mask, dim=0):
            mask = mask.to(grad.device)
            return apply_filter_binary_mask(mask, grad, dim=dim)

        for minfo in self.pruned_module_groups_info.get_all_nodes():
            mask = minfo.operand.binary_filter_pruning_mask
            weight = minfo.module.weight
            dim = minfo.module.target_weight_dim_for_compression
            partial_hook = update_wrapper(partial(hook, mask=mask, dim=dim), hook)
            self._hooks.append(weight.register_hook(partial_hook))
            if minfo.module.bias is not None:
                bias = minfo.module.bias
                partial_hook = update_wrapper(partial(hook, mask=mask), hook)
                self._hooks.append(bias.register_hook(partial_hook))

    def check_pruning_rate(self, params):
        """
        Check that set only one of pruning target params
        """
        pruning_target = params.get('pruning_target', None)
        pruning_flops_target = params.get('pruning_flops_target', None)
        if pruning_target and pruning_flops_target:
            raise ValueError('Only one parameter from \'pruning_target\' and \'pruning_flops_target\' can be set.')
        if pruning_flops_target:
            self.prune_flops = True

    def _clean_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_mask(self, minfo: PrunedModuleInfo) -> torch.Tensor:
        """
        Returns pruning mask for minfo.module.
        """
        raise NotImplementedError

    @staticmethod
    def pruning_rate_for_weight(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for all weight nodes.
        """
        weight = minfo.module.weight
        pruning_rate = 1 - weight.nonzero().size(0) / weight.view(-1).size(0)
        return pruning_rate

    @staticmethod
    def pruning_rate_for_filters(minfo: PrunedModuleInfo):
        """
        Calculates sparsity rate for weight filter-wise.
        """
        dim = minfo.module.target_weight_dim_for_compression
        weight = minfo.module.weight.transpose(0, dim).contiguous()
        filters_sum = weight.view(weight.size(0), -1).sum(axis=1)
        pruning_rate = 1 - len(filters_sum.nonzero()) / filters_sum.size(0)
        return pruning_rate

    def pruning_rate_for_mask(self, minfo: PrunedModuleInfo):
        mask = self.get_mask(minfo)
        pruning_rate = 1 - mask.nonzero().size(0) / max(mask.view(-1).size(0), 1)
        return pruning_rate

    def mask_shape(self, minfo: PrunedModuleInfo):
        mask = self.get_mask(minfo)
        return mask.shape

    def get_stats_for_pruned_modules(self):
        """
        Creates a table with layer pruning rate statistics
        """
        table = Texttable()
        table.set_cols_width([33, 20, 6, 8])
        header = ["Name", "Weight's shape", "Bias shape", "Layer PR"]
        data = [header]
        for minfo in self.pruned_module_groups_info.get_all_nodes():
            drow = {h: 0 for h in header}
            drow["Name"] = str(minfo.module_scope)
            drow["Weight's shape"] = list(minfo.module.weight.size())
            drow["Bias shape"] = list(minfo.module.bias.size()) if minfo.module.bias is not None else []
            drow["Layer PR"] = self.pruning_rate_for_filters(minfo)
            row = [drow[h] for h in header]
            data.append(row)
        table.add_rows(data)
        return table
