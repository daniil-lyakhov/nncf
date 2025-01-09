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
from copy import deepcopy
from typing import List, Tuple, Union

# This should be set befre any torch import
# to enable speed up for quantized model
# comipled with torch.compile.
os.environ["TORCHINDUCTOR_FREEZING"] = "1"

import argparse
import traceback
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.fx
from torch.jit import TracerWarning

from tests.torch.fx.performance_check import benchmark as b
from tests.torch.fx.performance_check import export as e
from tests.torch.fx.performance_check import quantization as q
from tests.torch.fx.performance_check.model_scope import MODEL_SCOPE
from tests.torch.fx.performance_check.model_scope import ModelConfig

warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class BenchmarkPipeline:
    def __init__(
        self,
        export_before_q: e.ExportInterface,
        compress: q.CompressionInterface,
        benchmarks: List[Tuple[e.ExportInterface, Union[List[b.BenchmarkInterface], b.BenchmarkInterface]]],
        enabled: bool = True,
    ):
        self.export_before_q = export_before_q
        self.compress = compress
        self.benchmarks = benchmarks
        self.enabled = enabled

    def run(self, model_name: torch.nn.Module, model_config: ModelConfig, save_dir: Path):
        if not self.enabled:
            return [], []

        pt_model = model_config.model_builder.build()

        exported_model = self.export_before_q(
            model=pt_model, model_config=model_config, path_to_save_model=save_dir / "ov_fp32_model.xml"
        )
        compressed_model = self.compress(exported_model, model_config, save_dir)
        prefix = [model_name, self.export_before_q.name(), self.compress.name(), self.compress.quantizer_name()]

        keys, values = [], []
        for export_after, benchmarks in self.benchmarks:
            compressed_model_path = save_dir / "ov_int8_model.xml"
            exported_compressed_model = export_after(
                model=deepcopy(compressed_model), model_config=model_config, path_to_save_model=compressed_model_path
            )
            benchmarks = benchmarks if isinstance(benchmarks, list) else [benchmarks]
            for benchmark in benchmarks:
                key = tuple(prefix + [export_after.name(), benchmark.name()])
                value = benchmark(exported_compressed_model, model_config, compressed_model_path)
                print("; ".join(key) + f" : {value}")
                keys.append(key)
                values.append(value)
        return keys, values


PIPELINES = (
    BenchmarkPipeline(e.NoExport(), q.NoQuantize(), [(e.NoExport(), b.LatencyBenchmark())], enabled=False),
    BenchmarkPipeline(
        e.TorchExport(),
        q.NoQuantize(),
        [
            (e.TorchCompileOVExport(), b.LatencyBenchmark()),
            (e.OpenvinoIRExport(), b.BenchmarkAppFPS(mode=b.BenchmarkAppMode.SYNC)),
        ],
    ),
    BenchmarkPipeline(
        e.TorchExport(),
        # CapturePreAutogradGraphExport(),
        q.NNCFQuantize(compress_weights=False),
        [
            (e.TorchCompileOVExport(), b.LatencyBenchmark()),
            # (OpenvinoIRExport(), LatencyBenchmark()),
            (e.OpenvinoIRExport(), b.BenchmarkAppFPS(mode=b.BenchmarkAppMode.SYNC)),
            # (OpenvinoIRExport(), BenchmarkAppFPS(mode=BenchmarkAppMode.ASYNC)),
        ],
    ),
    BenchmarkPipeline(
        e.TorchExport(),
        q.NNCFQuantize(compress_weights=True),
        [
            (e.TorchCompileOVExport(), b.LatencyBenchmark()),
            (e.OpenvinoIRExport(), b.BenchmarkAppFPS(mode=b.BenchmarkAppMode.SYNC)),
        ],
    ),
    BenchmarkPipeline(
        e.TorchExport(),
        q.NNCFQuantizePT2E(q.build_OpenVINOQuantizer, fold_quantize=False),
        [(e.TorchCompileOVExport(), b.LatencyBenchmark())],
    ),
    BenchmarkPipeline(
        e.TorchExport(),
        q.NNCFQuantizePT2E(q.build_X86Quantizer, fold_quantize=False),
        [(e.TorchCompileExport(), b.LatencyBenchmark())],
    ),
    BenchmarkPipeline(
        e.OpenvinoIRExport(),
        q.NNCFQuantize(compress_weights=True),
        [
            (e.NoExport(), b.LatencyBenchmark()),
            (e.NoExport(), b.BenchmarkAppFPS(mode=b.BenchmarkAppMode.SYNC)),
        ],
    ),
    BenchmarkPipeline(
        e.TorchExport(),
        q.TorchAOQuantize(q.build_XNNPACKQuantizer, fold_quantize=True),
        [(e.TorchCompileExport(), b.LatencyBenchmark())],
    ),
    BenchmarkPipeline(
        e.TorchExport(),
        q.NNCFQuantizePT2E(q.build_XNNPACKQuantizer, fold_quantize=True),
        [(e.TorchCompileExport(), b.LatencyBenchmark())],
    ),
)[
    -2:
]  # [4:6]  # [5:6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Target model name", type=str, default="all")
    parser.add_argument("--file_name", help="Output csv file_name", type=str, default="result.csv")

    args = parser.parse_args()

    target_models = []
    if args.model == "all":
        for model_name in MODEL_SCOPE:
            target_models.append(model_name)
    else:
        target_models.append(args.model)

    keys, values = [], []
    for model_name in target_models:
        print("---------------------------------------------------")
        print(f"name: {model_name}")
        try:
            model_config = MODEL_SCOPE[model_name]
            save_dir = Path(__file__).parent.resolve() / model_name
            save_dir.mkdir(exist_ok=True)
            for pipeline in PIPELINES:
                keys_, values_ = pipeline.run(model_name, model_config, save_dir)
                keys.extend(keys_)
                values.extend(values_)
        except Exception as e:
            print(f"FAILS TO CHECK PERFORMANCE FOR {model_name} MODEL:")
            err_msg = str(e)
            print(err_msg)
            traceback.print_exc()

    index = pd.MultiIndex.from_tuples(
        keys,
        names=["model", "export before int8", "compression", "quantizer", "export after int8", "benchmark key"],
    )
    df = pd.DataFrame(values, index=index)

    print(df)
    df.to_csv(args.file_name)


if __name__ == "__main__":
    main()
