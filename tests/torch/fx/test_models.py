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


from dataclasses import dataclass

import openvino.torch  # noqa
import pytest
import torch
import torch.fx
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch._export import capture_pre_autograd_graph

import nncf
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES
from tests.torch.fx.helpers import TinyImagenetDatasetManager

IMAGE_SIZE = 64
BATCH_SIZE = 128


@pytest.fixture(name="tiny_imagenet_dataset", scope="module")
def tiny_imagenet_dataset_fixture():
    return TinyImagenetDatasetManager(IMAGE_SIZE, BATCH_SIZE).create_data_loaders()


@dataclass
class ModelCase:
    model_id: str
    checkpoint_url: str


MODELS = (
    ModelCase(
        "resnet18",
        "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth",
    ),
)


def get_model(
    model_id: str, checkpoint_url: str, device: torch.device, num_classes: int = 200, in_features: int = 512
) -> torch.nn.Module:
    model = getattr(models, model_id)(weights=None)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    model.to(device)
    checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def getNodeType(node: torch.fx.node) -> str:
    if node.op == "call_function" and hasattr(node.target, "overloadpacket"):
        node_type = str(node.target).split(".")[1]
        return node_type
    return ""


def isNodeMetatype(node_type: str) -> bool:
    op_type = PT_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node_type)
    if op_type is UnknownMetatype:
        return False
    return True


def retrieve_nodes(model: torch.fx.GraphModule):
    for node in model.graph.nodes:
        yield node


@pytest.mark.parametrize("test_case", MODELS)
def test_sanity(test_case: ModelCase, tiny_imagenet_dataset):
    with disable_patching():
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = get_model(test_case.model_id, test_case.checkpoint_url, device)
        _, _, calibration_dataset = tiny_imagenet_dataset

        def transform_fn(data_item):
            return data_item[0].to(device)

        calibration_dataset = nncf.Dataset(calibration_dataset, transform_fn)

        with torch.no_grad():
            ex_input = next(iter(calibration_dataset.get_inference_data()))
            model.eval()
            exported_model = capture_pre_autograd_graph(model, args=(ex_input,))
            nodes = retrieve_nodes(exported_model)
            for node in nodes:
                node_type = getNodeType(node)
                if node_type:
                    assert isNodeMetatype(node_type)
