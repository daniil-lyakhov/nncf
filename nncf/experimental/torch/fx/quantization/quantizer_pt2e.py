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

import os

os.environ["TORCHINDUCTOR_FREEZING"] = "1"

from copy import deepcopy
from time import time
from typing import Any, Optional

import torch
import torch.fx
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager
from torchvision import models

import nncf
import nncf.torch
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.experimental.torch.fx.constant_folding import constant_fold
from nncf.experimental.torch.fx.quantization.fx_quantizer import NNCFFXQuantizer

# from nncf.experimental.torch.fx.transformations import apply_quantization_transformations
from nncf.experimental.torch.fx.transformations import fuse_conv_bn
from nncf.experimental.torch.fx.transformations import revert_quantization_transformations
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.scopes import IgnoredScope
from tests.torch.fx.helpers import visualize_fx_model


def measure_time(model, example_inputs, num_iters=3000):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for _ in range(num_iters):
            start_time = time()
            model(*example_inputs)
            total_time += time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def quantize_pt2e(
    model: torch.fx.GraphModule,
    quantizer: Any,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> torch.fx.GraphModule:
    """
    Implementation of the `quantize()` method for the Torch FX backend.
    """
    nncf_logger.warning(
        "Experimental Torch FX quantization backend is being used for the given torch.fx.GraphModule model."
        " Torch FX PTQ is an experimental feature, consider using Torch or OpenVino PTQ backends"
        " in case of errors or a poor model performance."
    )
    if target_device == TargetDevice.CPU_SPR:
        raise nncf.InternalError("target_device == CPU_SPR is not supported")
    if mode is not None:
        raise ValueError(f"mode={mode} is not supported")

    original_graph_meta = model.meta

    copied_model = deepcopy(model)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
        quantizer=NNCFFXQuantizer(quantizer),
    )

    # To make it easier for bias correction algorithms,
    # biases are being separated by the followng calls.
    anotated_model = deepcopy(copied_model)
    fuse_conv_bn(anotated_model)

    nncf_graph = NNCFGraphFactory.create(copied_model)
    quantized_model = quantization_algorithm.apply(copied_model, nncf_graph, dataset=calibration_dataset)

    # Revert applied transformation to keep original model
    # bias configuration.
    revert_quantization_transformations(quantized_model)

    # Magic. Without this call compiled model
    # is not preformant
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    quantized_model = _fold_conv_bn_qat(quantized_model)
    pm = PassManager([DuplicateDQPass()])

    quantized_model = pm(quantized_model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    quantized_model = pm(quantized_model).graph_module

    quantized_model.meta.update(original_graph_meta)
    quantized_model = _disallow_eval_train(quantized_model)

    return quantized_model


def main(model_cls):
    model = model_cls()
    example_inputs = torch.ones((1, 3, 224, 224))
    exported_model = capture_pre_autograd_graph(model.eval(), (example_inputs,))

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    nncf_quantizer_model = quantize_pt2e(exported_model, quantizer, calibration_dataset=nncf.Dataset([example_inputs]))

    visualize_fx_model(nncf_quantizer_model, "nncf_quantizer_before_fold_resnet.svg")
    return nncf_quantizer_model

    # exported_model = capture_pre_autograd_graph(model.eval(), (example_inputs,))
    # nncf_int8 = nncf.quantize(exported_model, nncf.Dataset([example_inputs]))
    # visualize_fx_model(nncf_int8, "nncf_resnet.svg")


def main_native(model_cls):
    model = model_cls()
    example_inputs = torch.ones((1, 3, 224, 224))
    exported_model = capture_pre_autograd_graph(model.eval(), (example_inputs,))

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    prepared_model(example_inputs)
    converted_model = convert_pt2e(prepared_model)
    visualize_fx_model(converted_model, "x86int8_resnet.svg")
    return converted_model


if __name__ == "__main__":
    with nncf.torch.disable_patching():
        for model_cls in (models.resnet18, models.mobilenet_v3_small, models.vit_b_16, models.swin_v2_s):
            # for model_cls in (models.mobilenet_v3_small,):
            print(f"{model_cls} check!")
            nncf_q_model = main(model_cls)

            constant_fold(nncf_q_model)
            visualize_fx_model(nncf_q_model, "nncf_quantizer_after_constant_fold_resnet.svg")

            pt_q_model = main_native(model_cls)
            print("benchmarking...")
            pt_compiled = torch.compile(model_cls())
            pt_int8_compiled = torch.compile(pt_q_model)
            nncf_comipled = torch.compile(nncf_q_model)

            example_inputs = (torch.ones((1, 3, 224, 224)),)

            pt_time = measure_time(pt_compiled, example_inputs)
            print(f"PT fp32 performance measured: {pt_time}")

            pt_int8_time = measure_time(pt_int8_compiled, example_inputs)
            print(f"PT int8 performance measured: {pt_int8_time}")

            nncf_int8_time = measure_time(nncf_comipled, example_inputs)
            print(f"NNCF int8 performance measured: {nncf_int8_time}")
