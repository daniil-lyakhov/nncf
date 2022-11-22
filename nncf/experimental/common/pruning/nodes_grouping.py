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
        self._group = None

    def split_by_reshape(self, shape_map):
        # TODO: make it common !!!
        if len(shape_map[1]) == 1:
            raise RuntimeError
        if len(shape_map[1]) > 2:
            raise NotImplementedError

        a = DimensionBlock(size=shape_map[1][-1], offset=0,
                           producer=self._producer,
                           opened_branches=self._opened_branches,
                           closed_branches=self._closed_branches)
        b = DimensionBlock(size=1, offset=shape_map[1][-1],
                           producer=self._producer,
                           opened_branches=self._opened_branches,
                           closed_branches=self._closed_branches)
        return [a, b]


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
        self._childs = []

    def get_actual_groups(self):
        if not self._childs:
            return self._blocks
        retval = []
        for child in self._childs:
            groups = child.get_actual_groups()
            retval.append(groups[0] if len(groups) == 1 else groups)
        return retval

    def has_childs(self):
        return bool(self._childs)

    def add_childs(self, childs):
        self._childs.extend(childs)

    def split_blocks_by_reshape(self, shape_map):
        if self._childs:
            raise NotImplementedError('Splitting BlockGroup with childs isn\'t implemented yet')

        new_blocks = []
        for block in self._blocks:
            new_blocks.append(block.split_by_reshape(shape_map))
        self._childs = []
        for group in zip(*new_blocks):
            self._childs.append(BlockGroup(list(group)))
        return self._childs.copy()

    # TODO: work on open branches
    def close_branch(self):
        for block in self._blocks:
            block.close_branch()

    def get_blocks(self):
        return self._blocks.copy()

    @staticmethod
    def join_groups(*args):
        for group in args:
            assert isinstance(group, BlockGroup), \
                f'Couldn\'t join args {args}, all elements should be BlockGroup instances'

        retval = BlockGroup([])
        for group in args:
            group.add_childs([retval])
            for block in group.get_blocks():
                retval._blocks.append(block)
        return retval


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
        self.block = None


def get_pruning_groups(graph: NNCFGraph,
                       pruning_operations_metatypes,
                       prune_operations_types):
    # 1. Initialize masks for producing nodes
    # TODO: clarify that all possibly pruned nodes will be collected here
    all_nodes_to_prune = graph.get_nodes_by_types(prune_operations_types)  # type: List[NNCFNode]
    roots = {}
    for node in all_nodes_to_prune:
        root_group = BlockGroup([DimensionBlock(MaskProducer(node.node_id))])
        roots[node.node_id] = root_group
        # TODO: make dimension map common here
        mask = PropagationMask({1: root_group})
        node.data['output_mask'] = mask

    # 2. Propagate masks
    MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()

    # 3. Collect groups from producers
    blocks_map = {}
    for id, group in roots.items():
        blocks_map[id] = group.get_actual_groups()

    return blocks_map
