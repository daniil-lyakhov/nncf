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
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTTransformationCommand
from nncf.torch.layer_utils import COMPRESSION_MODULES

COMPRESSION_STATE_ATTR = "compression_state"
EXAMPLE_INPUT_ATTR = "example_input"


class CompressionKeys(Enum):
    SHARED_INSERTION_COMMAND = "SHARED_INSERTION_COMMAND"
    INSERTION_COMMAND = "INSERTION_COMMAND"


def serialize_command(command: PTTransformationCommand):
    if not isinstance(command, (PTSharedFnInsertionCommand, PTInsertionCommand)):
        return {}

    serialized_transformation = dict()
    if isinstance(command, PTSharedFnInsertionCommand):
        serialized_transformation["type"] = CompressionKeys.SHARED_INSERTION_COMMAND.value
        serialized_transformation["target_points"] = [point.get_state() for point in command.target_points]
        serialized_transformation["op_name"] = command.op_name
        serialized_transformation["compression_module_type"] = command.compression_module_type

    elif isinstance(command, PTInsertionCommand):
        serialized_transformation["type"] = CompressionKeys.INSERTION_COMMAND.value
        serialized_transformation["target_point"] = command.target_point.get_state()

    # Check compression module is registered
    compression_module_name = command.fn.__class__.__name__
    if compression_module_name not in COMPRESSION_MODULES.registry_dict:
        raise RuntimeError(
            f"Could not serialize compression module with name {compression_module_name}."
            " Please register your module in the COMPRESSION_MODULES registry."
        )
    serialized_transformation["compression_module_name"] = compression_module_name
    serialized_transformation["fn_state"] = command.fn.get_state()
    serialized_transformation["hooks_group_name"] = command.hooks_group_name
    serialized_transformation["priority"] = command.priority.value
    return serialized_transformation


def serialize_transformations(model: torch.nn.Module, transformations_layout: TransformationLayout):
    transformation_commands = []
    for command in transformations_layout.transformations:
        serialized_command = serialize_command(command)
        if serialized_command:
            transformation_commands.append(serialized_command)

    return {
        COMPRESSION_STATE_ATTR: transformation_commands,
        EXAMPLE_INPUT_ATTR: model.nncf._input_info.get_forward_inputs(),
    }


def load_transformations(model: torch.nn.Module, transformations_state) -> torch.nn.Module:
    transformation_layout = TransformationLayout()
    for command in transformations_state[COMPRESSION_STATE_ATTR]:
        module_cls = COMPRESSION_MODULES.get(command["compression_module_name"])
        fn = module_cls.from_state(command["fn_state"])
        priority = TransformationPriority(command["priority"])
        if command["type"] == CompressionKeys.INSERTION_COMMAND.value:
            target_point = PTTargetPoint.from_state(command["target_point"])
            command = PTInsertionCommand(
                point=target_point, fn=fn, priority=priority, hooks_group_name=command["hooks_group_name"]
            )
            transformation_layout.register(command)
            continue

        if command["type"] == CompressionKeys.SHARED_INSERTION_COMMAND.value:
            target_points = [PTTargetPoint.from_state(state) for state in command["target_points"]]
            command = PTSharedFnInsertionCommand(
                target_points=target_points,
                fn=fn,
                op_unique_name=command["op_name"],
                compression_module_type=command["compression_module_type"],
                priority=priority,
                hooks_group_name=command["hooks_group_name"],
            )
            transformation_layout.register(command)

            continue
        raise RuntimeError(f"Command type {command['type']} is not supported.")
    return nncf.apply_transformations(model, transformation_layout, transformations_state[EXAMPLE_INPUT_ATTR])
