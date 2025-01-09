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
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Tuple

import openvino as ov
import torch
import torch.fx
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.quantizer import Quantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config
from torch.fx.passes.graph_drawer import FxGraphDrawer

import nncf
from nncf import AdvancedQuantizationParameters
from nncf.common.factory import NNCFGraphFactory
from nncf.experimental.quantization.quantizer.openvino_quantizer import OpenVINOQuantizer
from nncf.experimental.torch.fx.quantization.backend_parameters import FXBackendParameters
from nncf.experimental.torch.fx.quantization.quantize_pt2e import quantize_pt2e
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.performance_check.model_scope import ModelConfig

VISUALIZE_FX_INT8_GRAPH = True


class CompressionInterface(ABC):
    @abstractmethod
    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path):
        """
        Quantizes given model with given parameters
        """

    @abstractmethod
    def name(self) -> str:
        """
        The name of the quantization stage.
        """

    @abstractmethod
    def quantizer_name(self) -> str:
        """
        The name of the used quantizer.
        """


class NoQuantize(CompressionInterface):
    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> Any:
        return model

    def name(self) -> str:
        return "No quantization"

    def quantizer_name(self) -> str:
        return "-"


class NNCFQuantize(CompressionInterface):
    def __init__(self, compress_weights: bool, serialize_fx_int8_graph: bool = VISUALIZE_FX_INT8_GRAPH):
        self.serialize_fx_int8_graph = serialize_fx_int8_graph
        self.compress_weights = compress_weights

    @staticmethod
    def _get_backend(model) -> str:
        if isinstance(model, torch.fx.GraphModule):
            return "FX"
        if isinstance(model, ov.Model):
            return "OV"
        return ""

    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> Any:
        quantization_params = deepcopy(model_config.quantization_params)
        advanced_parameters = quantization_params.get("advanced_parameters", AdvancedQuantizationParameters())
        advanced_parameters.backend_params[FXBackendParameters.COMPRESS_WEIGHTS] = self.compress_weights
        quantization_params["advanced_parameters"] = advanced_parameters

        backend = self._get_backend(model)
        if "fx" in quantization_params:
            fx_params = quantization_params.pop("fx")
            if backend == "FX":
                quantization_params = {**quantization_params, **fx_params}

        with disable_patching():
            example_inputs = model_config.model_builder.get_example_inputs()
            quantized_model = nncf.quantize(
                model,
                nncf.Dataset(example_inputs),
                **quantization_params,
            )
        if backend == "OV":
            ov_int8_model_path = save_dir / "openvino_int8_model.xml"
            ov.serialize(quantized_model, ov_int8_model_path)
            print(f"Openvino quantized model saved to {ov_int8_model_path}")

        elif backend == "FX":
            _save_int8_torch_fx_info(
                quantized_model, save_dir, self.serialize_fx_int8_graph, f"nncf_compress_{self.compress_weights}"
            )

        int8_graph_visualization_path = str(
            save_dir / f"{backend}_int8_nncf_graph_compress_{self.compress_weights}.dot"
        )
        NNCFGraphFactory.create(quantized_model).visualize_graph(int8_graph_visualization_path)
        print(f"NNCFGraph visualization of int8 model is saved to {int8_graph_visualization_path}")

        return quantized_model

    def name(self) -> str:
        return f"nncf.quantize(compress_weights=={self.compress_weights})"

    def quantizer_name(self) -> str:
        return "-"


class TorchAOQuantize(CompressionInterface):
    def __init__(
        self,
        quantizer_builder: Callable[[Tuple[Any, ...]], Quantizer],
        fold_quantize: bool,
        serialize_fx_int8_graph: bool = VISUALIZE_FX_INT8_GRAPH,
    ) -> None:
        """
        fold_quantize == False for the torch.compile("openvino") inference
        """
        self.fold_quantize = fold_quantize
        self.serialize_fx_int8_graph = serialize_fx_int8_graph
        self.quantizer_builder = quantizer_builder

    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> torch.fx.GraphModule:
        assert isinstance(model, torch.fx.GraphModule)

        quantization_params = deepcopy(model_config.quantization_params)
        if "fx" in quantization_params:
            fx_params = quantization_params.pop("fx")
            quantization_params = {**quantization_params, **fx_params}

        quantizer = self.quantizer_builder(**quantization_params)

        with disable_patching():
            with torch.no_grad():
                example_inputs = model_config.model_builder.get_example_inputs()
                export_inputs = example_inputs[0] if isinstance(example_inputs[0], tuple) else example_inputs
                prepared_model = prepare_pt2e(model, quantizer)
                prepared_model(*export_inputs)
                quantized_model = convert_pt2e(prepared_model, fold_quantize=self.fold_quantize)
                _save_int8_torch_fx_info(
                    quantized_model,
                    save_dir,
                    self.serialize_fx_int8_graph,
                    f"torch_ao_{quantizer.__class__.__name__}_fold_{self.fold_quantize}",
                )
                return quantized_model

    def name(self) -> str:
        return f"torch.ao quantization quantizer: (fold_quantize=={self.fold_quantize})"

    def quantizer_name(self) -> str:
        return self.quantizer_builder.__name__.split("_")[-1]


class NNCFQuantizePT2E(CompressionInterface):
    def __init__(
        self,
        quantizer_builder: Callable[[Tuple[Any, ...]], Quantizer],
        fold_quantize: bool,
        serialize_fx_int8_graph: bool = VISUALIZE_FX_INT8_GRAPH,
    ) -> None:
        """
        fold_quantize == False for the torch.compile("openvino") inference
        """
        self.fold_quantize = fold_quantize
        self.serialize_fx_int8_graph = serialize_fx_int8_graph
        self.quantizer_builder = quantizer_builder

    def __call__(self, model: Any, model_config: ModelConfig, save_dir: Path) -> torch.fx.GraphModule:
        assert isinstance(model, torch.fx.GraphModule)

        quantization_params = deepcopy(model_config.quantization_params)
        if "fx" in quantization_params:
            fx_params = quantization_params.pop("fx")
            quantization_params = {**quantization_params, **fx_params}

        quantizer = self.quantizer_builder(**quantization_params)

        pt2e_kwargs = {}
        for key in (
            "subset_size",
            "fast_bias_correction",
            "smooth_quant",
            "bias_correction_params",
            "smooth_quant_params",
            "activations_range_estimator_params",
            "weights_range_estimator_params",
        ):
            if key in quantization_params:
                pt2e_kwargs[key] = quantization_params[key]
        smooth_quant = False
        if quantization_params.get("model_type", False):
            smooth_quant = quantization_params["model_type"] == nncf.ModelType.TRANSFORMER

        with disable_patching():
            example_inputs = model_config.model_builder.get_example_inputs()
            quantized_model = quantize_pt2e(
                model,
                quantizer,
                nncf.Dataset(example_inputs),
                smooth_quant=smooth_quant,
                fold_quantize=self.fold_quantize,
                **pt2e_kwargs,
            )
            _save_int8_torch_fx_info(
                quantized_model,
                save_dir,
                self.serialize_fx_int8_graph,
                f"nncf_pt2e_{quantizer.__class__.__name__}_fold_{self.fold_quantize}",
            )
            return quantized_model

    def name(self) -> str:
        return f"nncf.quantize_pt2e(fold_quantize=={self.fold_quantize})"

    def quantizer_name(self) -> str:
        return self.quantizer_builder.__name__.split("_")[-1]


def build_X86Quantizer(*args, **kwarsg) -> X86InductorQuantizer:
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())
    return quantizer


def build_XNNPACKQuantizer(*args, **kwargs) -> XNNPACKQuantizer:
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


def build_OpenVINOQuantizer(*args, **kwargs) -> OpenVINOQuantizer:

    quantizer_kwargs = {}
    for key in (
        "mode",
        "preset",
        "target_device",
        "model_type",
        "ignored_scope",
        "overflow_fix",
        "quantize_outputs",
        "activations_quantization_params",
        "weights_quantization_params",
        "quantizer_propagation_rule",
    ):
        if key in kwargs:
            quantizer_kwargs[key] = kwargs[key]
    return OpenVINOQuantizer(**quantizer_kwargs)


def _save_int8_torch_fx_info(
    quantized_model: torch.fx.GraphModule, save_dir: Path, serialize_fx_int8_graph: bool, q_backend: str
):
    int8_code_path = str(save_dir / f"int8_code_{q_backend}.py")
    with open(int8_code_path, "w") as f:
        f.write(quantized_model.code)
    print(f"int8 FX code is saved to {int8_code_path}")

    if serialize_fx_int8_graph:
        int8_model_visualization_path = str(save_dir / f"int8_fx_graph_q_backend_{q_backend}.svg")
        g = FxGraphDrawer(quantized_model, int8_model_visualization_path)
        g.get_dot_graph().write_svg(int8_model_visualization_path)
        print(f"Visualization of int8 model is saved to {int8_model_visualization_path}")
