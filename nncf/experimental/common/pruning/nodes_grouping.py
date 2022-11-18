from typing import List, Dict

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm


class DimensionBlock:
    def __init__(self, size: int, offset: int,
                 opened_branches: int = 0, closed_branches: int = 0) -> None:
        self.size = size
        self.offset = offset
        self.opened_branches = opened_branches
        self.closed_branches = closed_branches


class MaskProducer:
    def __init__(self, id_, blocks: List[DimensionBlock] = None) -> None:
        self.id = id_
        self.blocks = blocks
        if not blocks:
            self.blocks = [DimensionBlock(size=1, offset=0)]

    def split_block_by_reshape(self, block, shape_map):
        # TODO: make it common !!!
        if len(self.blocks) > 1:
            raise NotImplementedError
        for block_ in self.blocks:
            if block_ == block:
                if len(shape_map[1]) > 2:
                    raise NotImplementedError
                if len(shape_map[1]) == 1:
                    raise RuntimeError

                a = DimensionBlock(size=shape_map[1][-1], offset=0,
                                   opened_branches=block.opened_branches,
                                   closed_branches=block.closed_branches)
                b = DimensionBlock(size=1, offset=shape_map[1][-1],
                                   opened_branches=block.opened_branches,
                                   closed_branches=block.closed_branches)
                self.blocks = [a, b]


class PropagationMask:
    def __init__(self, producers: List[MaskProducer],
                 dim_block_map: Dict[DimensionBlock, int] = None):
        self.producers = producers
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
    producers = []
    for node in all_nodes_to_prune:
        producer = MaskProducer(node.node_id)
        producers.append(producer)
        # TODO: make dimention map common here
        mask = PropagationMask([producer], {1: producer.blocks[0]})
        node.data['output_mask'] = mask

    # 2. Propagate masks
    MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()

    # 3. Collect groups from producers
    pass
