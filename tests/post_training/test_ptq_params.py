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
from abc import abstractmethod

from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.onnx_backend import \
    ONNXMinMaxAlgoBackend

from tests.onnx.models import LinearModel
from tests.onnx.models import OneDepthwiseConvolutionalModel
from tests.post_training.models import NNCFGraphToTest


# pylint: disable=protected-access

@pytest.mark.parametrize('target_device', TargetDevice)
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device == target_device


class TemplateTestPTQParams:
    @abstractmethod
    def get_algo_backend(self):
        pass

    @abstractmethod
    def get_min_max_statistic_collector_cls(self):
        pass

    @abstractmethod
    def get_mean_max_statistic_collector_cls(self):
        pass

    @abstractmethod
    def check_quantize_outputs_fq_num(self, quantize_outputs,
                                      act_num_q, weight_num_q):
        pass

    @abstractmethod
    def check_ignored_scope_fq_num(self, ignored_scopes,
                                   act_num_q, weight_num_q):
        pass

    @abstractmethod
    @pytest.fixture
    def model_dict(self):
        pass

    @abstractmethod
    @pytest.fixture
    def ignored_scopes(self, request):
        pass

    @pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
    def test_range_type_per_tensor(self, model_dict, range_type):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        model = model_dict[self.test_range_type_per_tensor]
        assert min_max_algo._parameters.range_type == range_type
        stat_points = min_max_algo.get_statistic_points(model)

        for _, stat_point in stat_points.items():
            for stat_point_ in stat_point:
                for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                    if range_type is None:
                        # default tensor_collector for per-tensor
                        assert isinstance(tensor_collector, self.get_mean_max_statistic_collector_cls())
                    if range_type == RangeType.MINMAX:
                        assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())
                    elif range_type == RangeType.MEAN_MINMAX:
                        assert isinstance(tensor_collector, self.get_mean_max_statistic_collector_cls())

    @pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
    def test_range_type_per_channel(self, model_dict, range_type):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        model = model_dict[self.test_range_type_per_channel]
        assert min_max_algo._parameters.range_type == range_type
        stat_points = min_max_algo.get_statistic_points(model)

        for _, stat_point in stat_points.items():
            for stat_point_ in stat_point:
                for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                    # Range_type does not affect per-channel tensor_collector
                    assert isinstance(tensor_collector, self.get_min_max_statistic_collector_cls())

    @pytest.mark.parametrize('quantize_outputs', [False, True])
    def test_quantize_outputs(self, model_dict, quantize_outputs):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(quantize_outputs=quantize_outputs))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        nncf_graph = model_dict[self.test_quantize_outputs]
        assert min_max_algo._parameters.quantize_outputs == quantize_outputs
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
        act_num_q, weight_num_q = 0, 0
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                act_num_q += 1
            if quantization_point.is_weight_quantization_point():
                weight_num_q += 1

        self.check_quantize_outputs_fq_num(quantize_outputs,
                                           act_num_q, weight_num_q)

    def test_ignored_scopes(self, model_dict, ignored_scopes):
        algo = PostTrainingQuantization(PostTrainingQuantizationParameters(ignored_scopes=ignored_scopes))
        min_max_algo = algo.algorithms[0]
        min_max_algo._backend_entity = self.get_algo_backend()
        assert min_max_algo._parameters.ignored_scopes == ignored_scopes
        nncf_graph = model_dict[self.test_ignored_scopes]
        q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
        act_num_q, weight_num_q = 0, 0
        for quantization_point in q_setup.quantization_points.values():
            if quantization_point.is_activation_quantization_point():
                act_num_q += 1
            if quantization_point.is_weight_quantization_point():
                weight_num_q += 1

        self.check_ignored_scope_fq_num(ignored_scopes, act_num_q, weight_num_q)
