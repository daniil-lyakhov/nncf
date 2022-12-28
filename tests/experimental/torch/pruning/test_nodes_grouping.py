import torch
import numpy as np

from torch import nn

from nncf.torch import create_compressed_model
from nncf.torch.layers import NNCF_PRUNING_MODULES_DICT
from tests.torch.pruning.helpers import get_basic_pruning_config
from nncf.experimental.common.pruning.nodes_grouping import get_pruning_groups
from nncf.experimental.torch.pruning.operations import PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.o = nn.Linear(768, 768)
        self.sm = nn.Softmax()

    def forward(self, x):
        # assert x.shape == [384, 768]
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        k = k.view(-1, 12, 64).permute(1, 0, 2)
        q = q.view(-1, 12, 64).permute(1, 2, 0)
        v = v.view(-1, 12, 64).permute(1, 0, 2)
        x = self.sm(torch.matmul(k, q)) / np.sqrt(1 / 384)
        x = torch.matmul(x, v)
        x = x.permute(1, 0, 2).contiguous().view(-1, 768)
        return self.o(x)


def test_graph():
    config = get_basic_pruning_config([384, 768])
    config['compression']['algorithm'] = 'filter_pruning'
    model = SelfAttention()
    _, compression_ctrl = create_compressed_model(model, config)
    pruning_producing_types =  [x.op_func_name for x in NNCF_PRUNING_MODULES_DICT]
    blocks_map = get_pruning_groups(compression_ctrl .get_graph(),
                                PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES,
                                pruning_producing_types)
    print(blocks_map)
