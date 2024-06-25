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

import re
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import openvino as ov
import openvino.torch  # noqa
import torch
from torch._export import capture_pre_autograd_graph
from torch.export import Dim  # noqa
from torch.fx.passes.graph_drawer import FxGraphDrawer
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.models.yolo import YOLO
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel

import nncf

ROOT = Path(__file__).parent.resolve()


def measure_time(model, example_inputs, num_iters=500):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for i in range(0, num_iters):
            start_time = time.time()
            model(*example_inputs)
            total_time += time.time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def measure_time_ov(model, example_inputs, num_iters=1000):
    ie = ov.Core()
    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(example_inputs)
    total_time = 0
    for i in range(0, num_iters):
        start_time = time.time()
        infer_request.infer(example_inputs)
        total_time += time.time() - start_time
    average_time = (total_time / num_iters) * 1000
    return average_time


def validate_fx(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, num_samples: int = None
) -> Tuple[Dict, int, int]:
    # validator.seen = 0
    # validator.jdict = []
    # validator.stats = []
    # validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    for batch_i, batch in enumerate(data_loader):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        preds = model(batch["img"])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def print_statistics_short(stats: np.ndarray) -> None:
    mp, mr, map50, mean_ap = (
        stats["metrics/precision(B)"],
        stats["metrics/recall(B)"],
        stats["metrics/mAP50(B)"],
        stats["metrics/mAP50-95(B)"],
    )
    s = ("%20s" + "%12s" * 4) % ("Class", "Precision", "Recall", "mAP@.5", "mAP@.5:.95")
    print(s)
    pf = "%20s" + "%12.3g" * 4  # print format
    print(pf % ("all", mp, mr, map50, mean_ap))


def validate_ov(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, num_samples: int = None
) -> Tuple[Dict, int, int]:
    # validator.seen = 0
    # validator.jdict = []
    # validator.stats = []
    # validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
    model.reshape({0: [1, 3, -1, -1]})
    compiled_model = ov.compile_model(model)
    output_layer = compiled_model.output(0)
    for batch_i, batch in enumerate(data_loader):
        if num_samples is not None and batch_i == num_samples:
            break
        batch = validator.preprocess(batch)
        preds = torch.from_numpy(compiled_model(batch["img"])[output_layer])
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
    stats = validator.get_stats()
    return stats, validator.seen, validator.nt_per_class.sum()


def print_statistics(stats: np.ndarray, total_images: int, total_objects: int) -> None:
    mp, mr, map50, mean_ap = (
        stats["metrics/precision(B)"],
        stats["metrics/recall(B)"],
        stats["metrics/mAP50(B)"],
        stats["metrics/mAP50-95(B)"],
    )
    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Labels", "Precision", "Recall", "mAP@.5", "mAP@.5:.95")
    print(s)
    pf = "%20s" + "%12i" * 2 + "%12.3g" * 4  # print format
    print(pf % ("all", total_images, total_objects, mp, mr, map50, mean_ap))


def prepare_validation(model: YOLO, data: str) -> Tuple[Validator, torch.utils.data.DataLoader]:
    # custom = {"rect": True, "batch": 1}  # method defaults
    # rect: false forces to resize all input pictures to one size
    custom = {"rect": False, "batch": 1}  # method defaults
    args = {**model.overrides, **custom, "mode": "val"}  # highest priority args on the right

    validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
    stride = 32  # default stride
    validator.stride = stride  # used in get_dataloader() for padding
    validator.data = check_det_dataset(data)
    validator.init_metrics(de_parallel(model))

    data_loader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)

    return validator, data_loader


def benchmark_performance(model_path, config) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 30"
    command += f' -shape "[1,3,{config.imgsz},{config.imgsz}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec

    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def prepare_openvino_model(model: YOLO, model_name: str) -> Tuple[ov.Model, Path]:
    ir_model_path = Path(f"{ROOT}/{model_name}_openvino_model/{model_name}.xml")
    if not ir_model_path.exists():
        onnx_model_path = Path(f"{ROOT}/{model_name}.onnx")
        if not onnx_model_path.exists():
            model.export(format="onnx", dynamic=True, half=False)

        ov.save_model(ov.convert_model(onnx_model_path), ir_model_path)
    return ov.Core().read_model(ir_model_path), ir_model_path


def quantize(
    model: ov.Model, data_loader: torch.utils.data.DataLoader, validator: Validator, original_model
) -> ov.Model:
    def transform_fn(data_item: Dict):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader
        item for quantization.
        Parameters:
        data_item: Dict with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor

    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

    quantized_model = nncf.quantize(
        model,
        quantization_dataset,
        preset=nncf.QuantizationPreset.MIXED,
        ignored_scope=nncf.IgnoredScope(
            types=["Multiply", "Subtract", "Sigmoid"],
            subgraphs=[
                nncf.Subgraph(
                    inputs=["/model.22/Concat", "/model.22/Concat_1", "/model.22/Concat_2"],
                    outputs=["output0/sink_port_0"],
                )
            ],
        ),
    )
    return quantized_model


NNCF_QUANTIZATION = False


def quantize_impl(exported_model, val_loader, validator):
    def transform_fn(x):
        batch = validator.preprocess(x)
        return batch["img"]

    calibration_dataset = nncf.Dataset(val_loader, transform_fn)
    dir_name = str(Path(__file__).parent)
    if NNCF_QUANTIZATION:
        converted_model = nncf.quantize(
            exported_model,
            calibration_dataset,
            ignored_scope=nncf.IgnoredScope(
                types=["mul", "sub", "sigmoid"],
                subgraphs=[
                    nncf.Subgraph(
                        inputs=["cat_13", "cat_14", "cat_15"],
                        outputs=["output"],
                    )
                ],
            ),
        )
        g = FxGraphDrawer(converted_model, "yolo_nncf_fx_int8")
        g.get_dot_graph().write_svg(dir_name + "/yolo_nncf_fx_int8.svg")

        quantized_model = torch.compile(converted_model, backend="openvino")
        return quantized_model
    else:
        from torch.ao.quantization.quantize_pt2e import convert_pt2e
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e
        from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
        from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config

        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config())

        prepared_model = prepare_pt2e(exported_model, quantizer)

        for idx, batch in tqdm(enumerate(calibration_dataset.get_inference_data())):
            if idx >= 300:
                break
            prepared_model(batch)

        converted_model = convert_pt2e(prepared_model)

        g = FxGraphDrawer(prepared_model, "yolo_torch_fx_int8")
        g.get_dot_graph().write_svg(dir_name + "/yolo_torch_fx_int8.svg")
        import torch._inductor.config as config

        config.cpp_wrapper = True

        quantized_model = torch.compile(converted_model)
        return quantized_model


TORCH_FX = True
MODEL_NAME = "yolov8n"


def main():

    model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")

    # args = get_cfg(cfg=DEFAULT_CFG)
    # args.data = "coco128.yaml"
    # Prepare validation dataset and helper

    validator, data_loader = prepare_validation(model, "coco128.yaml")

    # Convert to OpenVINO model
    batch = next(iter(data_loader))
    batch = validator.preprocess(batch)

    if TORCH_FX:
        fp_stats, total_images, total_objects = validate_fx(model.model, tqdm(data_loader), validator)
        print("Floating-point Torch model validation results:")
        print_statistics(fp_stats, total_images, total_objects)

        if NNCF_QUANTIZATION:
            fp32_compiled_model = torch.compile(model.model, backend="openvino")
        else:
            fp32_compiled_model = torch.compile(model.model)
        fp32_stats, total_images, total_objects = validate_fx(fp32_compiled_model, tqdm(data_loader), validator)
        print("FP32 FX model validation results:")
        print_statistics(fp32_stats, total_images, total_objects)

        print("Start quantization...")
        # Rebuild model to reset ultralitics cache
        model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")
        with torch.no_grad():
            model.model.eval()
            model.model(batch["img"])
            # dynamic_shapes = ((None, None, Dim("H", min=1, max=29802), Dim("W", min=1, max=29802)),)
            dynamic_shapes = ((None, None, None, None),)
            exported_model = capture_pre_autograd_graph(
                model.model, args=(batch["img"],), dynamic_shapes=dynamic_shapes
            )
            quantized_model = quantize_impl(deepcopy(exported_model), data_loader, validator)

        int8_stats, total_images, total_objects = validate_fx(quantized_model, tqdm(data_loader), validator)
        print("INT8 FX model validation results:")
        print_statistics(int8_stats, total_images, total_objects)

        print("Start FX fp32 model benchmarking...")
        fp32_latency = measure_time(fp32_compiled_model, (batch["img"],))
        print(f"fp32 FX latency: {fp32_latency}")

        print("Start FX int8 model benchmarking...")
        int8_latency = measure_time(quantized_model, (batch["img"],))
        print(f"FX int8 latency: {int8_latency}")
        print(f"Speed up: {fp32_latency / int8_latency}")
        return

    ov_model, ov_model_path = prepare_openvino_model(model, MODEL_NAME)

    # Quantize mode in OpenVINO representation
    quantized_model = quantize(ov_model, data_loader, validator, model)
    quantized_model_path = Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}_quantized.xml")
    ov.save_model(quantized_model, str(quantized_model_path), compress_to_fp16=False)

    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128.yaml"
    # Validate FP32 model
    fp_stats, total_images, total_objects = validate_ov(ov_model, tqdm(data_loader), validator)
    print("Floating-point model validation results:")
    print_statistics(fp_stats, total_images, total_objects)

    # Validate quantized model
    q_stats, total_images, total_objects = validate_ov(quantized_model, tqdm(data_loader), validator)
    print("Quantized model validation results:")
    print_statistics(q_stats, total_images, total_objects)

    fps = True
    latency = True
    fp_model_perf = -1
    quantized_model_perf = -1
    if fps:
        # Benchmark performance of FP32 model
        fp_model_perf = benchmark_performance(ov_model_path, args)
        print(f"Floating-point model performance: {fp_model_perf} FPS")

        # Benchmark performance of quantized model
        quantized_model_perf = benchmark_performance(quantized_model_path, args)
        print(f"Quantized model performance: {quantized_model_perf} FPS")
    if latency:
        fp_model_latency = measure_time_ov(ov_model, batch["img"])
        print(f"FP32 OV model latency: {fp_model_latency}")
        int8_model_latency = measure_time_ov(quantized_model, batch["img"])
        print(f"INT8 OV model latency: {int8_model_latency}")

    return fp_stats["metrics/mAP50-95(B)"], q_stats["metrics/mAP50-95(B)"], fp_model_perf, quantized_model_perf


def main_export_not_strict():
    model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")

    # Prepare validation dataset and helper
    validator, data_loader = prepare_validation(model, "coco128.yaml")

    batch = next(iter(data_loader))
    batch = validator.preprocess(batch)

    model.model(batch["img"])
    ex_model = torch.export.export(model.model, args=(batch["img"],), strict=False)
    ex_model = capture_pre_autograd_graph(ex_model.module(), args=(batch["img"],))
    ex_model = torch.compile(ex_model)

    fp_stats, total_images, total_objects = validate_fx(ex_model, tqdm(data_loader), validator)
    print("Floating-point ex strict=False")
    print_statistics(fp_stats, total_images, total_objects)

    quantized_model = quantize_impl(deepcopy(ex_model), data_loader, validator)
    int8_stats, total_images, total_objects = validate_fx(quantized_model, tqdm(data_loader), validator)
    print("Int8 ex strict=False")
    print_statistics(int8_stats, total_images, total_objects)
    # No quantized were inserted, metrics are OK


if __name__ == "__main__":
    # main_export_not_strict()
    main()
