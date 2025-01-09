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

import re
import subprocess
from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, List, Tuple

import openvino as ov
import torch
import torch.fx

from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.performance_check.model_scope import ModelConfig


class BenchmarkInterface(ABC):
    @abstractmethod
    def __call__(self, model: Any, model_config: ModelConfig, model_path: Path) -> Any:
        """
        Benchmarks given model.
        """

    @abstractmethod
    def name(self) -> str:
        """
        Name of the Benchmarking stage.
        """


class LatencyBenchmark(BenchmarkInterface):
    def __call__(self, model: Any, model_config: ModelConfig, model_path: Path) -> Any:
        with disable_patching():
            with torch.no_grad():
                example_inputs = model_config.model_builder.get_example_inputs()
                if isinstance(model, ov.Model):
                    return measure_time_ov(model, example_inputs, model_config.num_iters)
                return measure_time(model, example_inputs, model_config.num_iters)

    def name(self) -> str:
        return "Latency, msec"


class BenchmarkAppMode(Enum):
    SYNC = "sync"
    ASYNC = "async"


class BenchmarkAppFPS(BenchmarkInterface):
    def __init__(self, mode: BenchmarkAppMode) -> None:
        self.mode = mode

    def __call__(self, model: Any, model_config: ModelConfig, model_path: Path) -> Any:
        fps, latency = benchmark_performance(
            model_path=model_path,
            input_shape=model_config.model_builder.get_input_sizes(),
            mode=self.mode.value,
            num_iters=model_config.num_iters,
        )
        return fps, latency

    def name(self) -> str:
        return f"Benchmark app: {self.mode.value} (FPS, latency, msec))"


def measure_time(model, example_inputs, num_iters=500):
    with torch.no_grad():
        model(*example_inputs)
        total_time = 0
        for _ in range(num_iters):
            start_time = time()
            model(*example_inputs)
            total_time += time() - start_time
        average_time = (total_time / num_iters) * 1000
    return average_time


def measure_time_ov(model, example_inputs, num_iters=500):
    ie = ov.Core()
    compiled_model = ie.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    infer_request.infer(example_inputs)
    total_time = 0
    for _ in range(num_iters):
        start_time = time()
        infer_request.infer(example_inputs)
        total_time += time() - start_time
    average_time = (total_time / num_iters) * 1000
    return average_time


def benchmark_performance(model_path: str, input_shape: List[int], mode: str, num_iters: int) -> Tuple[float, float]:
    if mode == "sync":
        exec_mode = "latency"
    else:
        exec_mode = "throughput"

    command = f"benchmark_app -m {model_path} -d CPU -hint {exec_mode} -niter {num_iters}"
    command += f' -shape "[{",".join(str(s) for s in input_shape)}]"'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec

    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    fps = float(match.group(1))

    match = re.search(r"Average\: (.+?) ms", str(cmd_output))
    latency = float(match.group(1))
    return fps, latency
