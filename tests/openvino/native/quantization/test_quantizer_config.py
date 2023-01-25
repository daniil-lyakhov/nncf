"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector

from tests.post_training.test_quantizer_config import TemplateTestQuantizerConfig


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return OVMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return OVMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return OVMeanMinMaxStatisticCollector
