from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype

from tests.post_training.models import NNCFGraphToTest
from tests.post_training.models import NNCFGraphToTestDepthwiseConv

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

import torch
from nncf.common.quantization.structs import QuantizationPreset
from nncf.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.data import Dataset
from nncf.parameters import convert_ignored_scope_to_list
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.model_creation import create_compressed_model
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.nested_objects_traversal import objwalk
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_tensor
from nncf.torch.dynamic_graph.graph_tracer import create_dummy_forward_fn
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo

from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm  import PostTrainingQuantizationParameters


def get_single_conv_nncf_graph() -> NNCFGraphToTest:
    conv_layer_attrs = ConvolutionLayerAttributes(
                            weight_requires_grad=True,
                            in_channels=4, out_channels=4, kernel_size=(4, 4),
                            stride=1, groups=1, transpose=False,
                            padding_values=[])
    return NNCFGraphToTest(PTModuleConv2dMetatype, conv_layer_attrs, PTNNCFGraph)


def get_depthwise_conv_nncf_graph() -> NNCFGraphToTestDepthwiseConv:
    return NNCFGraphToTestDepthwiseConv(PTDepthwiseConv2dSubtype)


def _create_nncf_config(input_shape):
    return NNCFConfig({
        'input_info': {
            'sample_size': input_shape
        }
    })


def get_nncf_network(model: torch.nn.Module,
                     input_shape: List[int] = [1, 3, 32, 32]):
    #input_shape = [1, 3, 32, 32]
    nncf_config = _create_nncf_config(input_shape)
    nncf_network = create_nncf_network(
        model=model,
        config=nncf_config,
    )
    return nncf_network
