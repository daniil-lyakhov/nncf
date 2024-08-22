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
from functools import partial
from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import openvino as ov
import torch
from fastdownload import FastDownload
from sklearn.metrics import accuracy_score
from torch._export import capture_pre_autograd_graph
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torchvision import datasets
from torchvision import models
from torchvision import transforms

import nncf
from nncf.common.logging.track_progress import track
from nncf.torch import disable_patching
from nncf.torch.dynamic_graph.patch_pytorch import unpatch_torch_operators

unpatch_torch_operators()
ROOT = Path(__file__).parent.resolve()
CHECKPOINT_URL = "https://huggingface.co/alexsu52/mobilenet_v2_imagenette/resolve/main/pytorch_model.bin"
DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATASET_PATH = "~/.cache/nncf/datasets"
DATASET_CLASSES = 10


def download_dataset() -> Path:
    downloader = FastDownload(base=DATASET_PATH, archive="downloaded", data="extracted")
    return downloader.get(DATASET_URL)


def load_checkpoint(model: torch.nn.Module) -> torch.nn.Module:
    checkpoint = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location=torch.device("cpu"), progress=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def validate(model: ov.Model, val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []
    with torch.no_grad():
        for images, target in track(val_loader, description="Validating"):
            pred = model(images)
            predictions.append(np.argmax(pred, axis=1))
            references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


def measure_latency(model, example_inputs, num_iters=1000):
    with torch.no_grad():
        model(example_inputs)
        total_time = 0
        for _ in range(num_iters):
            start_time = time()
            model(example_inputs)
            total_time += time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def get_model_size(ir_path: Path, m_type: str = "Mb", verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin")
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} {m_type}")
        print(f"Model weights (bin): {bin_size:.3f} {m_type}")
        print(f"Model size:          {model_size:.3f} {m_type}")
    return model_size


###############################################################################
# Create a PyTorch model and dataset

dataset_path = download_dataset()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    root=dataset_path / "val",
    transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
batch_size = 128
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

torch_model = models.mobilenet_v2(num_classes=DATASET_CLASSES)
torch_model = load_checkpoint(torch_model)
device = torch.device("cpu")
torch_model.to(device)
torch_model.eval()

###############################################################################
# Quantize a PyTorch model

# The transformation function transforms a data item into model input data.
#
# To validate the transform function use the following code:
# >> for data_item in val_loader:
# >>    model(transform_fn(data_item, device))


def transform_fn(data_item: Tuple[torch.Tensor, int], device: torch.device) -> torch.Tensor:
    images, _ = data_item
    return images.to(device)


# The calibration dataset is a small, no label, representative dataset
# (~100-500 samples) that is used to estimate the range, i.e. (min, max) of all
# floating point activation tensors in the model, to initialize the quantization
# parameters.

# The easiest way to define a calibration dataset is to use a training or
# validation dataset and a transformation function to remove labels from the data
# item and prepare model input data. The quantize method uses a small subset
# (default: 300 samples) of the calibration dataset.

# Recalculation default subset_size parameter based on batch_size.
subset_size = 300 // batch_size
calibration_dataset = nncf.Dataset(val_data_loader, partial(transform_fn, device=device))
dummy_input = torch.randn(1, 3, 224, 224).to(device)
fx_model = capture_pre_autograd_graph(torch_model.to(device), args=(dummy_input,))
fx_quantized_model = nncf.quantize(fx_model, calibration_dataset, subset_size=subset_size)


quantized_fx_svg_path = "mobilenet_v2_int8.svg"
FxGraphDrawer(fx_quantized_model, quantized_fx_svg_path).get_dot_graph().write_svg(quantized_fx_svg_path)

###############################################################################
# Benchmark performance, calculate compression rate and validate accuracy

compiled_model = torch.compile(torch_model, backend="openvino")
compiled_quantized_model = torch.compile(fx_quantized_model, backend="openvino")

print("[3/7] Benchmark FP32 model:")
with disable_patching():
    fp32_latency = measure_latency(compiled_model, dummy_input)
print(f"Latency: {fp32_latency}")

print("[4/7] Benchmark INT8 model:")
with disable_patching():
    int8_latency = measure_latency(compiled_quantized_model, dummy_input)
print(f"Latency: {int8_latency}")

print("[5/7] Validate compiled FP32 model:")
fp32_top1 = validate(compiled_model, val_data_loader)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[6/7] Validate compiled INT8 model:")
int8_top1 = validate(compiled_quantized_model, val_data_loader)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[7/7] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
# print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f"Performance speed up (throughput mode): {int8_latency / fp32_latency:.3f}")
