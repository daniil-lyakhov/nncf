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
from copy import deepcopy

import openvino as ov
import openvino.torch
import torch
from ultralytics import YOLO

import nncf

model = YOLO("yolo11n")

# Prepare validation dataset and helper
ex_input = torch.ones([1, 3, 640, 640])

model.model(ex_input)
ex_model = torch.export.export(model.model, args=(ex_input,), strict=False).module()

calibration_dataset = nncf.Dataset([ex_input])
quantized_model = nncf.quantize(
    ex_model,
    calibration_dataset,
    preset=nncf.QuantizationPreset.MIXED,
    model_type=nncf.ModelType.TRANSFORMER,
    ignored_scope=nncf.IgnoredScope(
        types=["mul", "sub", "sigmoid", "__getitem__"],
        subgraphs=[
            nncf.Subgraph(
                inputs=["cat_13", "cat_14", "cat_15"],
                outputs=["output"],
            )
        ],
    ),
)


quantized_model = torch.export.export(quantized_model, args=(ex_input,))
ov_model = ov.convert_model(deepcopy(quantized_model), example_input=ex_input)
compiled_model = ov.Core().compile_model(ov_model)

# Works as expected
for _ in range(3):
    compiled_model(ex_input)


quantized_model = torch.compile(
    quantized_model.module(),
    backend="openvino",
)

# Raises an error
for _ in range(3):
    quantized_model(ex_input)
