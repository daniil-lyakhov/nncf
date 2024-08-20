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

import argparse
import copy
import time
import warnings
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Type

import numpy as np
import openvino.torch  # noqa
import pandas as pd
import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torchvision.models as models
from sklearn.metrics import accuracy_score
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.jit import TracerWarning
from torchvision import datasets

import nncf
from nncf import QuantizationPreset
from nncf.common.logging.track_progress import track
from nncf.parameters import ModelType
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# DATASET_IMAGENET = "/home/dlyakhov/datasets/imagenet/val"
DATASET_IMAGENET = "/home/dlyakhov/datasets/imagenet_one_pic/imagenet/val"


@dataclass
class ModelConfig:
    model_cls: Type[torch.nn.Module]
    model_weights: models.WeightsEnum
    quantization_params: Dict[str, Any]


MODELS_DICT = {
    "vit_b_16": ModelConfig(models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, {"model_type": ModelType.TRANSFORMER}),
    "swin_v2_s": ModelConfig(models.swin_v2_s, models.Swin_V2_S_Weights.DEFAULT, {"model_type": ModelType.TRANSFORMER}),
    "resnet50": ModelConfig(models.resnet50, models.ResNet50_Weights.DEFAULT),
    "mobilenet_v3_small": ModelConfig(
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.DEFAULT,
        {"preset": QuantizationPreset.MIXED, "fast_bias_correction": False},
    ),
}


def measure_time(model, example_inputs, num_iters=1000):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for i in range(0, num_iters):
            start_time = time.time()
            model(*example_inputs)
            total_time += time.time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def quantize(model, example_inputs, calibration_dataset, subset_size=300):
    with torch.no_grad():
        exported_model = capture_pre_autograd_graph(model, example_inputs)

    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported_model, quantizer)
    from tqdm import tqdm

    for inp, _ in islice(tqdm(calibration_dataset), subset_size):
        prepared_model(inp)
    converted_model = convert_pt2e(prepared_model)
    return converted_model


def validate(model, val_loader, subset_size=None):
    dataset_size = len(val_loader)

    predictions = np.zeros((dataset_size))
    references = -1 * np.ones((dataset_size))

    with track(total=dataset_size, description="Validation") as pbar:

        for i, (images, target) in enumerate(val_loader):
            if subset_size is not None and i >= subset_size:
                break

            output_data = model(images).detach().numpy()
            predicted_label = np.argmax(output_data, axis=1)
            predictions[i] = predicted_label.item()
            references[i] = target
            pbar.progress.update(pbar.task, advance=1)
    acc_top1 = accuracy_score(predictions, references) * 100
    print(acc_top1)
    return acc_top1


def fx_2_ov_quantization(pt_model, example_input, output_dir, result, val_loader, shape_input):
    with disable_patching():
        fp32_pt_model = copy.deepcopy(pt_model)
        fp32_compile_model = torch.compile(fp32_pt_model, backend="openvino")

        quant_pt_model = quantize(fp32_compile_model, (example_input,), val_loader)
        quant_compile_model = torch.compile(quant_pt_model, backend="openvino")

        g = FxGraphDrawer(quant_pt_model, f"b_pt_{pt_model.__class__.__name__}_int8")
        g.get_dot_graph().write_svg(f"b_pt_{pt_model.__class__.__name__}_int8.svg")

        acc1_quant_model = validate(quant_compile_model, val_loader)
        result["acc1_quant_model"] = acc1_quant_model

        latency_fx = measure_time(quant_compile_model, (example_input,))
        print(f"latency: {latency_fx}")
        result["torch_compile_latency_fps_quant_model"] = latency_fx


def process_model(model_name: str):
    result = {"name": model_name}

    model_config = MODELS_DICT[model_name]

    ##############################################################
    # Prepare dataset
    ##############################################################

    val_dataset = datasets.ImageFolder(root=DATASET_IMAGENET, transform=model_config.model_weights.transforms())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

    ##############################################################
    # Prepare original model
    ##############################################################

    pt_model = model_config.model_cls(weights=model_config.model_weights)
    pt_model = pt_model.eval()
    example_input = next(iter(val_loader))[0]

    ##############################################################
    # Process FP32 Model
    ##############################################################

    orig_acc1 = model_config.model_weights.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")
    result["fp32_acc@1"] = orig_acc1
    print(f"fp32 model metric: {orig_acc1}")

    latency_fp32 = measure_time(torch.compile(pt_model, backend="openvino"), (example_input,))
    result["fp32_latency"] = latency_fp32
    print(f"fp32 model latency: {latency_fp32}")

    with disable_patching():
        with torch.no_grad():
            exported_model = capture_pre_autograd_graph(pt_model, (example_input,))

    def transform(x):
        return x[0]

    quant_fx_model = nncf.quantize(
        exported_model, nncf.Dataset(val_loader, transform_func=transform), **model_config.quantization_params
    )
    quant_fx_model = torch.compile(quant_fx_model, backend="openvino")

    int8_model_visualization_path = f"{pt_model.__class__.__name__}_int8.svg"
    g = FxGraphDrawer(quant_fx_model, int8_model_visualization_path)
    g.get_dot_graph().write_svg(int8_model_visualization_path)
    print(f"Visualization of int8 model is saved to {int8_model_visualization_path}.")

    int8_acc1 = validate(quant_fx_model, val_loader)
    result["int8_acc@1"] = int8_acc1
    print(f"int8 model metric: {int8_acc1}")

    latency_int8 = measure_time(quant_fx_model, (example_input,))
    result["int8_latency"] = latency_int8
    print(f"int8 model metric: {latency_int8}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="torchvision model name", type=str, default="all")
    parser.add_argument("--file_name", help="output csv file_name", type=str, default="result.csv")

    args = parser.parse_args()

    results_list = []
    if args.model == "all":
        for model_name in MODELS_DICT:
            print("---------------------------------------------------")
            print(f"name: {model_name}")
            results_list.append(process_model(model_name))
    else:
        results_list.append(process_model(args.model))

    df = pd.DataFrame(results_list)
    print(df)
    df.to_csv(args.file_name)


if __name__ == "__main__":
    main()
