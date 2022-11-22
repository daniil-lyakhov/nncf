from typing import Type, List, Optional
from collections import defaultdict

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import NNCFGraph
from nncf.common.tensor import NNCFTensor
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import identity_mask_propagation
from nncf.experimental.common.pruning.nodes_grouping import PropagationMask
from nncf.experimental.common.pruning.nodes_grouping import BlockGroup


class BasePruningOp:
    """
    Determines meta operations which aggregate operations having common
    properties of interaction with pruning masks
    """

    subtypes = []
    additional_types = []

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        """
        Propagates the pruning mask through a node using pruning masks of all inputs and the current node (if any).

        :param node: The graph node to propagate mask through it
        :param graph: The model graph to prune
        :param tensor_processor: Interface with tensor processing methods
        """
        raise NotImplementedError

    @classmethod
    def get_all_op_aliases(cls) -> List[str]:
        """
        :return: list of all aliases of types in metatype
        """
        op_types = []
        for subtype in cls.subtypes:
            op_types.extend(subtype.get_all_aliases())
        op_types = list(set(op_types)) + cls.additional_types
        return op_types


class InputPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class OutputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class IdentityMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)

class ConvolutionPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)
        # TODO: make dimentionwise mask propagation

        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class TransposeConvolutionPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        output_mask = node.data.get('output_mask', None)

        # TODO: make dimentionwise mask propagation
        # In case of group convs we can't prune by output filters
        if is_grouped_conv(node):
            output_mask = None
            if is_prunable_depthwise_conv(node):
                output_mask = input_masks[0]

        node.data['output_mask'] = output_mask


class LinearPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) in [1, 2]
        if len(input_masks) == 1 and input_masks[0] is not None:
            output_mask = node.data['output_mask']
            # Propagating batch dims
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            input_shape_len = len(input_tensors_shapes[0])
            for dim, block in input_masks[0].dim_block_map.items():
                if dim ==  input_shape_len - 1:
                    blocks = block if isinstance(block, list) else [block]
                    for block in blocks:
                        block.close_branch()
                else:
                    output_mask.dim_block_map[dim] = block

        elif len(input_masks) == 2:
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            assert len(input_tensors_shapes[0]) == len(input_tensors_shapes[1])
            input_shape_len = len(input_tensors_shapes[0])
            # Join consumed masks
            # TODO: Consider that input tensors are in the right order
            left_dim_block, right_dim_block = [input_masks[i].dim_block_map for i in range(2)]
            def _both_dim_blocks_exist(left_idx, right_idx):
                if left_idx in left_dim_block or \
                    right_idx in right_dim_block:
                    assert left_idx in left_dim_block and \
                        right_idx in right_dim_block
                    return True
                return False

            output_mask = PropagationMask()
            # Propagating batch dims
            for dim in range(input_shape_len - 2):
                if _both_dim_blocks_exist(dim, dim):
                   output_mask.dim_block_map[dim] = BlockGroup.join_groups(left_dim_block[dim],
                                                                           right_dim_block[dim])
            # Propagating left rows / right cols
            for idx, dim in enumerate(range(input_shape_len - 2, input_shape_len)):
                if dim in input_masks[idx].dim_block_map:
                    output_mask.dim_block_map[dim] = input_masks[idx].dim_block_map[dim]

            # Close branch
            if _both_dim_blocks_exist(input_shape_len - 1, input_shape_len - 2):
                left = left_dim_block[input_shape_len - 1]
                right = right_dim_block[input_shape_len - 2]
                BlockGroup.join_groups(left, right).close_branch()

        else:
            output_mask = node.data.get('output_mask', None)

        node.data['output_mask'] = output_mask


class BatchNormPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
               and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        if cls.accept_pruned_input(node):
            identity_mask_propagation(node, graph)
        else:
            node.data['output_mask'] = None


class ConcatPruningOp(BasePruningOp):
    @classmethod
    def generate_output_mask(cls, node: NNCFNode, graph: NNCFGraph) -> Optional[NNCFTensor]:
        """
        Generate output mask from input masks with all None replaced by identity masks.
        If all input masks is None return None.

        :param node: Node to determine it's sources.
        :param graph: NNCF graph to work with.
        :param tensor_processor: Interface with tensor processing methods.
        :return: Filled input masks.
        """
        input_edges = graph.get_input_edges(node)
        previous_nodes = [edge.from_node for edge in input_edges]
        input_masks = [input_node.data['output_mask'] for input_node in previous_nodes]

        not_empty_masks = [mask for mask in input_masks if mask is not None]  # type: List[NNCFTensor]
        if not not_empty_masks:
            return None

        first_non_empty_mask = not_empty_masks[0]
        device = first_non_empty_mask.device
        filled_input_masks = []
        for i, mask in enumerate(input_masks):
            if mask is None:
                concat_axis = node.layer_attributes.axis
                concat_dim = input_edges[i].tensor_shape[concat_axis]
                mask = tensor_processor.ones(concat_dim, device)
            filled_input_masks.append(mask)
        result_mask = tensor_processor.concatenate(filled_input_masks, 0)
        return result_mask

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        result_mask = cls.generate_output_mask(node, graph, tensor_processor)
        node.data['output_mask'] = result_mask


class ElementwisePruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        #TODO: Make eltwise common
        # This case is for true div
        identity_mask_propagation(node, graph)
        return
        input_masks = get_input_masks(node, graph)
        output_mask = input_masks[0]
        if output_mask is not None:
            output_mask = tensor_processor.elementwise_mask_propagation(input_masks)

        node.data['output_mask'] = output_mask


class ReshapePruningOp(BasePruningOp):
    @staticmethod
    def _is_flatten(node: NNCFNode) -> bool:
        return len(node.layer_attributes.output_shape) == 2

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is None:
            return False
        return True

    @staticmethod
    def _map_dims(node: NNCFNode) -> bool:
        def _check_dim_splitted(dim_from: int, dims_to: List[int],
                                dims_to_start_idx: int):
            idx = dims_to_start_idx
            accum = dims_to[idx]
            while accum < dim_from:
                idx += 1
                accum *= dims_to[idx]
            if accum > dim_from:
                return (False, idx)
            return (True, idx)

        def _map_dims_(dims_from: List[int], dims_to: List[int],
                      from_idx: int, to_idx: int, from_map, to_map):
            res, to_idx_next = _check_dim_splitted(dims_from[from_idx], dims_to, to_idx)
            if not res:
                return (res, to_idx_next)
            from_map[from_idx] = list(range(to_idx, to_idx_next + 1))
            for idx in range(to_idx, to_idx_next + 1):
                to_map[idx] = from_idx
            return (res, to_idx_next)

        input_shape = node.layer_attributes.input_shape
        output_shape = node.layer_attributes.output_shape

        inp_idx = 0
        out_idx = 0
        inp_map = {}
        out_map = {}

        mode = 'default'

        while (inp_idx < len(input_shape) and out_idx < len(output_shape)):
            if input_shape[inp_idx] == output_shape[out_idx]:
                inp_map[inp_idx] = out_idx
                out_map[out_idx] = inp_idx
            elif input_shape[inp_idx] > output_shape[out_idx]:
                res, out_idx = _map_dims_(input_shape, output_shape,
                                         inp_idx, out_idx, inp_map, out_map)
                if not res or mode == 'shrink':
                    return None
                mode = 'extend'
            else:
                res, out_idx = _map_dims_(output_shape, input_shape,
                                         out_idx, inp_idx, out_map, inp_map)
                if not res or mode == 'extend':
                    return None
                mode = 'shrink'
            inp_idx += 1
            out_idx += 1
        return inp_map, out_map, mode

    @classmethod
    def _get_propagated_mask(cls, node: NNCFNode, graph: NNCFGraph):
        if not node.layer_attributes:
            return None

        map = cls._map_dims(node)
        if not map:
            return None

        inp_map, out_map, mode = map
        input_shape = node.layer_attributes.input_shape
        output_shape = node.layer_attributes.output_shape

        masks = get_input_masks(node, graph)
        assert len(masks) == 1
        mask = masks[0]
        if mode == 'default':
            return mask

        output_mask = PropagationMask()
        if mode == 'extend':
            map = mask.dim_block_map
            for dim, group in mask.dim_block_map.items():
                if isinstance(group, list):
                    raise NotImplementedError('Extend reshape for several groups is not supported yet')
                shape_map = [input_shape[dim], [output_shape[x] for x in inp_map[dim]]]
                new_groups = group.split_blocks_by_reshape(shape_map)
                for new_group, in_dim in zip(new_groups, inp_map[dim]):
                    output_mask.dim_block_map[in_dim] = new_group
            return output_mask

        if mode == 'shrink':
            grouping = defaultdict(list)
            for inp_idx, groups in mask.dim_block_map.items():
                groups = groups if isinstance(groups, list) else [groups]
                grouping[inp_map[inp_idx]].extend(groups)
            output_mask.dim_block_map = dict(grouping)
            return output_mask


    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = cls._get_propagated_mask(node, graph)


class TransposePruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        new_order = node.layer_attributes.permutation
        idx_map = [(old_idx, new_idx) for new_idx, old_idx in enumerate(new_order) if old_idx in input_mask.dim_block_map]
        output_mask = PropagationMask({new_idx: input_mask.dim_block_map[old_idx] for old_idx, new_idx in idx_map})

        node.data['output_mask'] = output_mask


class FlattenPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is not None:
            return True
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]):
        output_mask = None
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is not None and node.layer_attributes is not None:
            # We assume all layers known by the mask propagation algo except
            # StopMaskForwardOp have input/output batch dim == 0.
            # Besides, since input_mask is not None thus no StopMaskForwardOp operations
            # was in the path from mask producer node to this node. As all
            # known nodes have input/output batch dim == 0 previous has too.
            flatten_channels = node.layer_attributes.output_shape[1]
            mask_len = input_mask.shape[0]
            assert flatten_channels % mask_len == 0
            output_mask = tensor_processor.repeat(input_mask, repeats=flatten_channels // mask_len)

        node.data['output_mask'] = output_mask


class StopMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None
