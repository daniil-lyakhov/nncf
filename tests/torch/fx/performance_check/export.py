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

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import openvino as ov
import openvino.torch  # noqa
import torch
import torch.fx
from torch._export import capture_pre_autograd_graph

from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.performance_check.model_scope import ModelConfig


class ExportInterface(ABC):
    @abstractmethod
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        """
        Converts passed torch.nn.Module to the target representation
        """

    @abstractmethod
    def name(self) -> str:
        """
        Return name of the export before quantization stage.
        """


class NoExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        return model

    def name(self) -> str:
        return "No export"


class CapturePreAutogradGraphExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> torch.fx.GraphModule:
        with disable_patching():
            with torch.no_grad():
                return capture_pre_autograd_graph(model, args=model_config.model_builder.get_example_inputs())

    def name(self) -> str:
        return "capture_pre_autograd_graph"


class TorchExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        with disable_patching():
            with torch.no_grad():
                return torch.export.export(
                    model, args=model_config.model_builder.get_example_inputs(), strict=model_config.torch_export_strict
                ).module()

    def name(self) -> str:
        return "torch.export.export"


class OpenvinoIRExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path) -> Any:
        with disable_patching():
            with torch.no_grad():
                example_inputs = model_config.model_builder.get_example_inputs()
                export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
                input_sizes = model_config.model_builder.get_input_sizes()
                ex_model = torch.export.export(model, export_inputs)
                ov_model = ov.convert_model(ex_model, example_input=example_inputs[0], input=input_sizes)
                ov.serialize(ov_model, path_to_save_model)
                return ov_model

    def name(self) -> str:
        return "Export to openvino IR"


class TorchCompileExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path):
        return torch.compile(model)

    def name(self) -> str:
        return "torch.compile(...)"


class TorchCompileOVExport(ExportInterface):
    def __call__(self, model: Any, model_config: ModelConfig, path_to_save_model: Path):
        return torch.compile(model, backend="openvino")

    def name(self) -> str:
        return "torch.compile(..., backend='openvino')"
