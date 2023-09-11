# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Dict, Tuple, Type

import pytest
import torch

from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.collectors import PTNNCFCollectorTensorProcessor
from nncf.torch.tensor_statistics.collectors import get_mean_percentile_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_median_mad_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_min_max_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_mixed_min_max_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_precentile_tensor_collector
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic


class TestCollectedStatistics:
    REF_INPUTS = [
        torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[4.5, 2.6, 3.7], [-1.3, -4, -3.5], [4.3, 5.8, 6.1]]),
    ]

    @pytest.mark.parametrize(
        ("collector", "reduction_shapes_vs_ref_statistic"),
        [
            (
                get_min_max_statistic_collector,
                {
                    ((1,), (0, 1)): PTMinMaxTensorStatistic(
                        {"min_values": torch.tensor([-4.0]), "max_values": torch.tensor([6.1])}
                    ),
                    ((3, 1), (1,)): PTMinMaxTensorStatistic(
                        {
                            "min_values": torch.tensor([[1.0], [-4.0], [4.0]]),
                            "max_values": torch.tensor([[4.5], [4.0], [6.1]]),
                        }
                    ),
                    ((1, 3), (0,)): PTMinMaxTensorStatistic(
                        {
                            "min_values": torch.tensor([[-1.3, -4.0, -3.5]]),
                            "max_values": torch.tensor([[4.5, 5.8, 6.1]]),
                        }
                    ),
                    # Not supported for now:
                    # ((3, 3), ): PTMinMaxTensorStatistic(
                    #     min_values=torch.tensor([
                    #         [1.0, 2.0, 3.0],
                    #         [-1.3, -4, -3.5],
                    #         [4.0, 5.0, 6.0]
                    #     ]),
                    #     max_values=torch.tensor([
                    #         [4.5, 2.6, 3.7],
                    #         [1.3, 4.0, 3.5],
                    #         [4.3, 5.8, 6.1]
                    #     ]),
                    # ),
                },
            ),
            (
                partial(
                    get_mixed_min_max_statistic_collector,
                    use_means_of_mins=True,
                    use_means_of_maxs=True,
                ),
                {
                    ((1,), (0, 1)): PTMinMaxTensorStatistic(
                        {"min_values": torch.tensor([-3.5]), "max_values": torch.tensor([6.05])}
                    ),
                    ((3, 1), (1,)): PTMinMaxTensorStatistic(
                        {
                            "min_values": torch.tensor([[1.8], [-3.5], [4.15]]),
                            "max_values": torch.tensor([[3.75], [3.5], [6.05]]),
                        }
                    ),
                    ((1, 3), (0,)): PTMinMaxTensorStatistic(
                        {
                            "min_values": torch.tensor([[-1.15, -3, -3.25]]),
                            "max_values": torch.tensor([[4.25, 5.4, 6.05]]),
                        }
                    ),
                },
            ),
            (
                partial(
                    get_mixed_min_max_statistic_collector,
                    use_means_of_mins=False,
                    use_means_of_maxs=True,
                ),
                {
                    ((1,), (0, 1)): PTMinMaxTensorStatistic(
                        {"min_values": torch.tensor([-4.0]), "max_values": torch.tensor([6.05])}
                    ),
                    ((3, 1), (1,)): PTMinMaxTensorStatistic(
                        {
                            "min_values": torch.tensor([[1.0], [-4.0], [4.0]]),
                            "max_values": torch.tensor([[3.75], [3.5], [6.05]]),
                        }
                    ),
                    ((1, 3), (0,)): PTMinMaxTensorStatistic(
                        {
                            "min_values": torch.tensor([[-1.3, -4.0, -3.5]]),
                            "max_values": torch.tensor([[4.25, 5.4, 6.05]]),
                        }
                    ),
                },
            ),
        ],
    )
    def test_collected_statistics_with_shape_convert(
        self,
        collector: Type[TensorStatisticCollectorBase],
        reduction_shapes_vs_ref_statistic: Dict[Tuple[ReductionShape, ReductionShape], TensorStatistic],
    ):
        for shapes in reduction_shapes_vs_ref_statistic.keys():
            output_shape, reducer_axes = shapes
            collector_obj = collector(
                use_abs_max=True,
                reducers_axes=reducer_axes,
                reducers_keepdims=len(output_shape) > 1,
                aggregators_axes=(0,),
                aggregators_keepdims=False,
                squeeze_dims=None,
                num_samples=None,
            )
            for input_ in TestCollectedStatistics.REF_INPUTS:
                collector_obj.register_unnamed_inputs(PTNNCFTensor(input_))
            test_stats = collector_obj.get_statistics()
            assert reduction_shapes_vs_ref_statistic[shapes] == test_stats

    @pytest.mark.parametrize(
        ("collector", "reduction_shapes_vs_ref_statistic"),
        [
            (
                get_median_mad_statistic_collector,
                # PTMedianMADStatisticCollector,
                {
                    (1,): PTMedianMADTensorStatistic(
                        {
                            "tensor_statistic_output": {
                                "median_values": torch.tensor([2.8]),
                                "mad_values": torch.tensor([2.6]),
                            }
                        }
                    ),
                    (3, 1): PTMedianMADTensorStatistic(
                        {
                            "tensor_statistic_output": {
                                "median_values": torch.tensor([[2.8], [-2.5], [5.4]]),
                                "mad_values": torch.tensor([[0.85], [1.1], [0.65]]),
                            }
                        }
                    ),
                    (1, 3): PTMedianMADTensorStatistic(
                        {
                            "tensor_statistic_output": {
                                "median_values": torch.tensor([[2.5, 2.3, 3.35]]),
                                "mad_values": torch.tensor([[1.9, 3.1, 2.7]]),
                            }
                        }
                    ),
                    # Not supported for now:
                    # (3, 3): PTMedianMADTensorStatistic(
                    #     median_values=torch.tensor([
                    #         [2.75, 2.3, 3.35],
                    #         [-1.15, -3, -3.25],
                    #         [4.15, 5.4, 6.05]
                    #     ]),
                    #     mad_values=torch.tensor([
                    #         [1.75, 0.3, 0.35],
                    #         [0.15, 1, 0.25],
                    #         [0.15, 0.4, 0.05]
                    #     ]),
                    # ),
                },
            ),
            (
                partial(get_precentile_tensor_collector, percentiles_to_collect=[10.0]),
                # partial(PTPercentileStatisticCollector, percentiles_to_collect=[10.0]),
                {
                    (1,): PTPercentileTensorStatistic({"tensor_statistic_output": {10.0: torch.tensor([-3.15])}}),
                    (3, 1): PTPercentileTensorStatistic(
                        {"tensor_statistic_output": {10.0: torch.tensor([[1.5], [-3.75], [4.15]])}}
                    ),
                    (1, 3): PTPercentileTensorStatistic(
                        {"tensor_statistic_output": {10.0: torch.tensor([[-1.15, -3, -3.25]])}}
                    ),
                    # Not supported for now:
                    # (3, 3): PTPercentileTensorStatistic(
                    #     {
                    #         10.0: torch.tensor([
                    #             [1.35, 2.06, 3.07],
                    #             [-1.27, -3.8, -3.45],
                    #             [4.03, 5.08, 6.01]
                    #         ])
                    #     }
                    # ),
                },
            ),
            (
                # partial(PTMeanPercentileStatisticCollector, percentiles_to_collect=[10.0]),
                partial(get_mean_percentile_statistic_collector, percentiles_to_collect=[10.0]),
                {
                    (1,): PTPercentileTensorStatistic({"tensor_statistic_output": {10.0: torch.tensor([-2.9])}}),
                    (3, 1): PTPercentileTensorStatistic(
                        {"tensor_statistic_output": {10.0: torch.tensor([[2.0100], [-3.3500], [4.4000]])}}
                    ),
                    (1, 3): PTPercentileTensorStatistic(
                        {"tensor_statistic_output": {10.0: torch.tensor([[-0.3900, -1.9400, -1.9300]])}}
                    ),
                    # Not supported for now:
                    # (3, 3): PTPercentileTensorStatistic(
                    #     {
                    #         10.0: torch.tensor([
                    #             [ 2.7500,  2.3000,  3.3500],
                    #             [-1.1500, -3.0000, -3.2500],
                    #             [ 4.1500,  5.4000,  6.0500]
                    #         ])
                    #     }
                    # ),
                },
            ),
        ],
    )
    def test_collected_statistics(
        self,
        collector: Type[TensorStatisticCollectorBase],
        reduction_shapes_vs_ref_statistic: Dict[ReductionShape, TensorStatistic],
    ):
        for reduction_shape in reduction_shapes_vs_ref_statistic:
            if len(reduction_shape) > 1:
                reducer_axes = ([dim for dim, val in enumerate(reduction_shape) if val == 1][0],)
                aggregator_keep_dims = False
            else:
                reducer_axes = (0, 1)
                aggregator_keep_dims = True

            collector_obj = collector(
                reducers_axes=reducer_axes,
                reducers_keepdims=len(reduction_shape) > 1,
                aggregators_axes=(0,),
                aggregators_keepdims=aggregator_keep_dims,
                num_samples=None,
                squeeze_dims=None,
            )
            for input_ in TestCollectedStatistics.REF_INPUTS:
                if hasattr(collector_obj, "register_unnamed_inputs"):
                    collector_obj.register_unnamed_inputs(PTNNCFTensor(input_))
                else:
                    collector_obj.register_inputs(input_)
            test_stats = collector_obj.get_statistics()
            assert reduction_shapes_vs_ref_statistic[reduction_shape] == test_stats


class TestCollectorTensorProcessor:
    tensor_processor = PTNNCFCollectorTensorProcessor()

    def test_unstack(self):
        # Unstack tensor with dimensions
        tensor1 = torch.tensor([1.0])
        tensor_unstacked1 = TestCollectorTensorProcessor.tensor_processor.unstack(PTNNCFTensor(tensor1))

        # Unstack dimensionless tensor
        tensor2 = torch.tensor(1.0)
        tensor_unstacked2 = TestCollectorTensorProcessor.tensor_processor.unstack(PTNNCFTensor(tensor2))

        assert tensor_unstacked1 == tensor_unstacked2 == [PTNNCFTensor(torch.tensor(1.0))]
