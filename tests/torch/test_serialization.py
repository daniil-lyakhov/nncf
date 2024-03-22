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

import functools

import pytest

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.serialization import load_command
from nncf.torch.graph.transformations.serialization import serialize_command
from nncf.torch.layer_utils import COMPRESSION_MODULES
from tests.torch.helpers import DummyOpWithState
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import check_commands_are_equal


@pytest.mark.parametrize(
    "target_type",
    (
        TargetType.OPERATION_WITH_WEIGHTS,
        TargetType.OPERATOR_PRE_HOOK,
        TargetType.OPERATOR_POST_HOOK,
        TargetType.PRE_LAYER_OPERATION,
        TargetType.POST_LAYER_OPERATION,
    ),
)
@pytest.mark.parametrize(
    "command_builder",
    (
        TwoConvTestModel.create_pt_insertion_command,
        functools.partial(
            TwoConvTestModel.create_pt_shared_fn_insertion_command,
            compression_module_type=ExtraCompressionModuleType.EXTERNAL_OP,
        ),
        functools.partial(
            TwoConvTestModel.create_pt_shared_fn_insertion_command,
            compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
        ),
    ),
)
@pytest.mark.parametrize(
    "priority", (TransformationPriority.QUANTIZATION_PRIORITY, TransformationPriority.QUANTIZATION_PRIORITY.value + 1)
)
def test_serialize_load_command(target_type, command_builder, priority):
    group_name = "CUSTOM_HOOKS_GROUP_NAME"
    dummy_op_state = "DUMMY_OP_STATE"

    if DummyOpWithState.__name__ not in COMPRESSION_MODULES.registry_dict:
        registered_dummy_op_cls = COMPRESSION_MODULES.register()(DummyOpWithState)
    else:
        registered_dummy_op_cls = DummyOpWithState
    dummy_op = registered_dummy_op_cls(dummy_op_state)
    command = command_builder(target_type, priority, fn=dummy_op, group=group_name)

    if isinstance(command, PTSharedFnInsertionCommand) and target_type in [
        TargetType.PRE_LAYER_OPERATION,
        TargetType.POST_LAYER_OPERATION,
    ]:
        pytest.skip(f"PTSharedFnInsertionCommand is not supporting target type {target_type}")

    serialized_command = serialize_command(command)
    recovered_command = load_command(serialized_command)
    check_commands_are_equal(recovered_command, command, check_fn_ref=False)

    assert isinstance(command.fn, DummyOpWithState)
    assert command.fn.get_state() == recovered_command.fn.get_state() == dummy_op_state
