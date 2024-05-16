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
import time
import warnings
from copy import deepcopy
from typing import Dict

import numpy as np
import onnx
import onnxruntime as rt
import torch
from common import QUANTIZED_CHECKPOINT_FILE_NAME
from common import ROOT
from common import get_data_loader
from common import get_mobilenet_v2
from common import validate
from torch.jit import TracerWarning

import nncf.torch
from nncf.common.utils.helpers import create_table

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ONNXEngine:
    """
    Engine for ONNX backend using ONNXRuntime to infer the model.
    """

    def __init__(self, model, **rt_session_options):
        self.input_names = set()
        rt_session_options["providers"] = ["OpenVINOExecutionProvider"]
        serialized_model = model.SerializeToString()
        self.sess = rt.InferenceSession(serialized_model, **rt_session_options)

        for inp in self.sess.get_inputs():
            self.input_names.add(inp.name)

    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via ONNXRuntime InferenceSession.
        Returns the dictionary of model outputs by node names.
        :param input_data: inputs for the model
        :return output_data: models outputs
        """
        self.sess.run([], {k: v for k, v in input_data.items() if k in self.input_names})


def run_benchmark(path_to_model, shape, verbose):
    onnx_model = onnx.load(path_to_model)
    engine = ONNXEngine(onnx_model)
    inputs = {name: np.ones(shape).astype(np.float32) for name in engine.input_names}
    start = time.time()
    for _ in range(10000):
        engine.infer(inputs)
    duration = time.time() - start
    if verbose:
        print(f"Inference took {duration} seconds.")
    return duration


###############################################################################
# Recover the quantized model, benchmark performance, calculate compression rate and validate accuracy

device = torch.device("cpu")
torch_model = get_mobilenet_v2(device)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

quantized_checkpoint_path = ROOT / QUANTIZED_CHECKPOINT_FILE_NAME
print(f"[1/8] Recover INT8 PyTorch model model from the checkpoint: {quantized_checkpoint_path }")
if not os.path.isfile(quantized_checkpoint_path):
    raise RuntimeError(
        f"File {quantized_checkpoint_path} is not found."
        " Please quantize the model first by running quantize.py script."
    )

quantized_checkpoint = torch.load(quantized_checkpoint_path)
torch_quantized_model = nncf.torch.load_from_config(
    deepcopy(torch_model), quantized_checkpoint["nncf_config"], dummy_input
)
torch_quantized_model.load_state_dict(quantized_checkpoint["model_state_dict"])

fp32_model_path = ROOT / "mobilenet_v2_fp32.onnx"
torch.onnx.export(torch_model, dummy_input, fp32_model_path)
print(f"[2/8] Save FP32 model: {fp32_model_path}")

int8_model_path = ROOT / "mobilenet_v2_int8.onnx"
torch_quantized_model = nncf.strip(torch_quantized_model)
torch.onnx.export(torch_quantized_model, dummy_input, int8_model_path)
print(f"[3/8] Save INT8 model: {int8_model_path}")

print("[4/8] Benchmark FP32 model:")
fp32_fps = run_benchmark(fp32_model_path, shape=[1, 3, 224, 224], verbose=True)
print("[5/8] Benchmark INT8 model:")
int8_fps = run_benchmark(int8_model_path, shape=[1, 3, 224, 224], verbose=True)

val_loader = get_data_loader(1)

print("[6/8] Validate ONNX FP32 model in OpenVINO:")
fp32_top1 = validate(fp32_model_path, val_loader)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[7/8] Validate ONNX INT8 model in OpenVINO:")
int8_top1 = validate(int8_model_path, val_loader)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[8/8] Report:")
tabular_data = [
    [
        "Accuracy@1",
        fp32_top1,
        int8_top1,
        f"Accuracy drop: {fp32_top1 - int8_top1:.3f}",
    ],
    ["Performance, FPS", fp32_fps, int8_fps, f"Speedup x{int8_fps / fp32_fps:.3f}"],
]
print(create_table(["", "FP32", "INT8", "Summary"], tabular_data))
