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

import pytest

from nncf.parameters import TargetDevice
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.onnx_backend import \
    ONNXMinMaxAlgoBackend

from tests.onnx.models import LinearModel
from tests.onnx.models import OneDepthwiseConvolutionalModel
from tests.post_training.test_ptq_params import TemplateTestPTQParams
from tests.post_training.models import NNCFGraphToTest


# pylint: disable=protected-access

@pytest.mark.parametrize('target_device', TargetDevice)
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device == target_device


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return ONNXMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return ONNXMeanMinMaxStatisticCollector

    def check_quantize_outputs_fq_num(self, quantize_outputs,
                                      act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 2
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    def check_ignored_scope_fq_num(self, ignored_scopes,
                                   act_num_q, weight_num_q):
        if ignored_scopes:
            assert act_num_q == 0
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    @pytest.fixture
    def model_dict(self):
        return {self.test_range_type_per_tensor:
            LinearModel().onnx_model,
        self.test_range_type_per_channel:
            OneDepthwiseConvolutionalModel().onnx_model,
        self.test_quantize_outputs:
            NNCFGraphToTest().nncf_graph,
        self.test_ignored_scopes:
            NNCFGraphToTest().nncf_graph,
        }

    @pytest.fixture(params=[[], ['/Conv_1_0']])
    def ignored_scopes(self, request):
        return request.param
