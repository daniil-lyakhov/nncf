from typing import List, Dict

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm


class DimensionBlock:
    def __init__(self,
                 producer,
                 size: int = 1, offset: int = 0,
                 opened_branches: int = 0, closed_branches: int = 0) -> None:
        self.size = size
        self.offset = offset
        self._producer = producer
        self._opened_branches = opened_branches
        self._closed_branches = closed_branches
        self._childs = []
        self._group = None

    def split_by_reshape(self, shape_map):
        # TODO: make it common !!!
        if len(shape_map[1]) == 1:
            raise RuntimeError
        if len(shape_map[1]) > 2:
            raise NotImplementedError
        if self._childs:
            raise NotImplementedError

        a = DimensionBlock(size=shape_map[1][-1], offset=0,
                           producer=self._producer,
                           opened_branches=self._opened_branches,
                           closed_branches=self._closed_branches)
        b = DimensionBlock(size=1, offset=shape_map[1][-1],
                           producer=self._producer,
                           opened_branches=self._opened_branches,
                           closed_branches=self._closed_branches)

        self._childs.extend([a, b])

    def get_childs(self):
        return self._childs.copy()

    def open_branch(self):
        self._opened_branches += 1

    def close_branch(self):
        self._closed_branches += 1

    def set_group(self, group):
        self._group = group


class BlockGroup:
    def __init__(self, blocks) -> None:
        self._blocks = blocks # type: DimensionBlock
        for block in blocks:
            block.set_group(self)

    def split_blocks_by_reshape(self, shape_map):
        new_blocks = []
        for block in self._blocks:
            block.split_by_reshape(shape_map)
            new_blocks.append(block.get_childs())
        retval = []
        for group in zip(*new_blocks):
            retval.append(BlockGroup(list(group)))
        return retval

    # TODO: work on open branches
    def close_branch(self):
        for block in self._blocks:
            block.close_branch()

    def get_blocks(self):
        return self._blocks.copy()

    @staticmethod
    def join_groups(*args):
        blocks_union = []
        for group in args:
            for block in group.get_blocks():
                blocks_union.append(block)
        return BlockGroup(blocks_union)


class MaskProducer:
    def __init__(self, id_)  -> None:
        self.id = id_


class PropagationMask:
    def __init__(self,
                 dim_block_map: Dict[DimensionBlock, int] = None):
        self.dim_block_map = dim_block_map if dim_block_map is not None else {}


class PruningNodeGroup:
    def __init__(self) -> None:
        self.producing_nodes = []
        self.adjusted_nodes = []
        self.closing_nodes = []


def get_pruning_groups(graph: NNCFGraph,
                       pruning_operations_metatypes,
                       prune_operations_types):
    # 1. Initialize masks for producing nodes
    # TODO: clarify that all possibly pruned nodes will be collected here
    all_nodes_to_prune = graph.get_nodes_by_types(prune_operations_types)  # type: List[NNCFNode]
    roots = []
    for node in all_nodes_to_prune:
        root_block = DimensionBlock(MaskProducer(node.node_id))
        roots.append(root_block)
        # TODO: make dimension map common here
        mask = PropagationMask({1: BlockGroup([root_block])})
        node.data['output_mask'] = mask

    # 2. Propagate masks
    MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()

    # 3. Collect groups from producers
    pass
