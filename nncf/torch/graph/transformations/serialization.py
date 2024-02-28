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

from enum import Enum

import torch

import nncf
from examples.torch.common.model_loader import COMPRESSION_STATE_ATTR
from examples.torch.common.model_loader import MODEL_STATE_ATTR
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.model_transformer import PTQuantizerInsertionCommand
from nncf.torch.model_transformer import PTSharedFnInsertionCommand
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import PTQuantizerSpec


class CompressionKeys(Enum):
    QUANTIZER_INSERTION_COMMAND = "QUANTIZER_INSERTION_COMMAND"
    SHARED_INSERTION_COMMAND = "SHARED_INSERTION_COMMAND"


def serialize_transformations(model: torch.nn.Module, transformations_layout: TransformationLayout):
    transformation_commands = []
    for transformation in transformations_layout.transformations:
        if not isinstance(transformation, (PTQuantizerInsertionCommand, PTSharedFnInsertionCommand)):
            continue

        serialized_transformation = dict()
        if isinstance(transformation, PTQuantizerInsertionCommand):
            serialized_transformation["type"] = CompressionKeys.QUANTIZER_INSERTION_COMMAND.value
            serialized_transformation["target_point"] = transformation.target_point.get_state()
            serialized_transformation["quantizer_spec"] = transformation.quantizer.quantizer_spec.get_state()
        if isinstance(transformation, PTSharedFnInsertionCommand):
            serialized_transformation["type"] = CompressionKeys.SHARED_INSERTION_COMMAND.value
            serialized_transformation["target_points"] = [point.get_state() for point in transformation.target_points]
            serialized_transformation["fn_name"] = transformation.fn.__name__
            serialized_transformation["fn_state"] = transformation.fn.get_state()
            serialized_transformation["op_name"] = transformation.op_name
            serialized_transformation["priority"] = transformation.priority.value
        serialized_transformation["hooks_group_name"] = transformation.hooks_group_name
        transformation_commands.append(serialized_transformation)

    return {MODEL_STATE_ATTR: model.state_dict(), COMPRESSION_STATE_ATTR: transformation_commands}


def load_transformations(model: torch.nn.Module, transformations, example_input) -> torch.nn.Module:
    transformation_layout = TransformationLayout()
    for command in transformations[COMPRESSION_STATE_ATTR]:
        if command["type"] == CompressionKeys.QUANTIZER_INSERTION_COMMAND.value:
            qspec = PTQuantizerSpec.from_state(command["quantizer_spec"])
            quantizer_cls = QUANTIZATION_MODULES.get(qspec.mode)
            quantizer = quantizer_cls(qspec)
            target_point = PTTargetPoint.from_state(command["target_point"])
            command = PTQuantizerInsertionCommand(
                point=target_point, quantizer=quantizer, hooks_group_name=command["hooks_group_name"]
            )
            transformation_layout.register(command)
            continue

        if command["type"] == CompressionKeys.SHARED_INSERTION_COMMAND.value:
            target_points = [PTTargetPoint.from_state(state) for state in command["target_points"]]
            module_cls = COMPRESSION_MODULES.get(command["fn_name"])
            fn = module_cls.from_state(command["fn_state"])
            priority = TransformationPriority[command["priority"]]
            command = PTSharedFnInsertionCommand(
                target_points=target_points,
                fn=fn,
                op_unique_name=command["op_name"],
                priority=priority,
                hooks_group_name=command["hooks_group_name"],
            )
            transformation_layout.register(command)

            continue
        raise RuntimeError(f"Command type {command['type']} is not supported.")
    transformed_model = nncf.apply_transformations(model, transformation_layout, example_input)
    transformed_model.load_state_dict(transformations[MODEL_STATE_ATTR])
    return transformed_model
