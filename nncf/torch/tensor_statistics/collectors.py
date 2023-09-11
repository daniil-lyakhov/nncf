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
from typing import Any, Callable, Deque, List, Optional, Tuple, Union

import numpy as np
import torch

from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import MaskedReduceFN
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import NNCFTensor
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MedianAbsoluteDeviationAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import PostAggregateHook
from nncf.experimental.common.tensor_statistics.collectors import PrecentileAggregator
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.statistics import PTMeanTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic


# pylint: disable=too-many-public-methods
class PTNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for PTNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple, list], keepdims: bool = False) -> NNCFTensor:
        return PTNNCFTensor(torch.amin(x.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple, list], keepdims: bool = False) -> NNCFTensor:
        return PTNNCFTensor(torch.amax(x.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.abs(x.tensor))

    @classmethod
    def min(cls, *args) -> NNCFTensor:
        stacked = cls.stack(args)
        return cls.reduce_min(stacked, axis=0, keepdims=False)

    @classmethod
    def max(cls, *args) -> NNCFTensor:
        stacked = cls.stack(args)
        return cls.reduce_max(stacked, axis=0, keepdims=False)

    @staticmethod
    def mean(x: NNCFTensor, axis: Union[int, tuple, list], keepdims=False) -> NNCFTensor:
        return PTNNCFTensor(x.tensor.mean(dim=axis, keepdim=keepdims))

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, tuple, list], keepdims=False) -> NNCFTensor:
        # See https://github.com/pytorch/pytorch/issues/61582
        if not isinstance(axis, int):
            return PTNNCFTensor(torch.tensor(np.median(x.tensor.detach().cpu().numpy(), axis=axis, keepdims=keepdims)))
        return PTNNCFTensor(x.tensor.median(dim=axis, keepdim=keepdims).values)

    @classmethod
    def masked_mean(cls, x: NNCFTensor, axis: Union[int, tuple], mask: NNCFTensor, keepdims=False) -> NNCFTensor:
        if mask is None:
            return cls.mean(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor.detach().cpu().numpy(), mask=mask.tensor)
        result = np.ma.mean(masked_x, axis=axis, keepdims=keepdims).astype(masked_x.dtype)
        if result.size <= 1:
            return PTNNCFTensor(torch.tensor(result))
        return PTNNCFTensor(torch.tensor(result.data))

    @classmethod
    def masked_median(
        cls, x: NNCFTensor, axis: Union[int, tuple, list], mask: NNCFTensor, keepdims=False
    ) -> NNCFTensor:
        # Implemented in numy as torch.masked.median is not implemented yet
        if mask is None:
            return cls.median(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor.detach().cpu().numpy(), mask=mask.tensor.detach().cpu().numpy())
        result = np.ma.median(masked_x, axis=axis, keepdims=keepdims).astype(masked_x.dtype)
        if len(result) == 1:
            return PTNNCFTensor(torch.tensor(result))
        return PTNNCFTensor(torch.tensor(result.data))

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return PTNNCFTensor(torch.mean(x.tensor, axis=0))
        x = torch.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return PTNNCFTensor(torch.mean(t, axis=(0, 2)))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return PTNNCFTensor(torch.stack(x, dim=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        tensor = x.tensor
        if list(tensor.shape) == []:  # pylint: disable=C1803
            tensor = tensor.unsqueeze(0)
        tensor_list = torch.unbind(tensor, dim=axis)
        return [PTNNCFTensor(t) for t in tensor_list]

    @staticmethod
    def squeeze(x: NNCFTensor, dim: Optional[int] = None) -> NNCFTensor:
        return PTNNCFTensor(torch.squeeze(x.tensor, dim=dim))

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return torch.sum(tensor.tensor).item()

    @staticmethod
    def quantile(
        tensor: NNCFTensor, quantile: Union[float, List[float]], axis: Union[int, tuple, list], keepdims: bool = False
    ) -> List[NNCFTensor]:
        # See https://github.com/pytorch/pytorch/issues/61582
        if not isinstance(axis, int):
            result = torch.tensor(
                np.quantile(tensor.tensor.detach().cpu().numpy(), q=quantile, axis=axis, keepdims=keepdims)
            )
        else:
            result = torch.quantile(tensor.tensor, torch.tensor(quantile).type(tensor.tensor.dtype), axis, keepdims)
        result = result.type(tensor.tensor.dtype)
        return [PTNNCFTensor(x) for x in result]

    @classmethod
    def precentile(
        cls,
        tensor: NNCFTensor,
        precentile: Union[float, List[float]],
        axis: Union[int, tuple, list],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        quantile = np.true_divide(precentile, 100)
        return cls.quantile(tensor, quantile=quantile, axis=axis, keepdims=keepdims)

    @classmethod
    def no_outliers_map(
        cls,
        x: NNCFTensor,
        fn: Callable[[NNCFTensor, int, NNCFTensor], Any],
        axis: Union[int, Tuple[int, ...]] = 0,
        alpha: float = 0.01,
        keepdims: bool = False,
    ):
        if isinstance(axis, int):
            axis = (axis,)

        if len(x.shape) == len(axis):
            return fn(x, axis=axis, mask=None, keepdims=keepdims)

        low_values, high_values = cls.quantile(x, [alpha, 1 - alpha], axis=axis)
        outliers_mask = torch.logical_or(x.tensor < low_values.tensor, high_values.tensor < x.tensor)
        return fn(x, axis=axis, mask=PTNNCFTensor(outliers_mask), keepdims=keepdims)

    @classmethod
    def masked_map(cls, x: NNCFTensor, fn: MaskedReduceFN, filter_fn) -> NNCFTensor:
        return fn(x, mask=filter_fn(x))

    @classmethod
    def sub(cls, a: NNCFTensor, b: NNCFTensor) -> NNCFTensor:
        return NNCFTensor(a.tensor - b.tensor)

    @classmethod
    def zero_elements(cls, x: NNCFTensor) -> NNCFTensor:
        pt_tensor = x.tensor
        eps = torch.finfo(pt_tensor.dtype).eps
        return NNCFTensor(pt_tensor.abs() < eps)


class PTReducerMixIn:
    def _get_processor(self):
        return PTNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class PTNoopReducer(PTReducerMixIn, NoopReducer):
    pass


class PTMinReducer(PTReducerMixIn, MinReducer):
    pass


class PTMaxReducer(PTReducerMixIn, MaxReducer):
    pass


class PTAbsMaxReducer(PTReducerMixIn, AbsMaxReducer):
    pass


class PTMeanReducer(PTReducerMixIn, MeanReducer):
    pass


class PTQuantileReducer(PTReducerMixIn, QuantileReducer):
    pass


class PTAbsQuantileReducer(PTReducerMixIn, AbsQuantileReducer):
    pass


class PTBatchMeanReducer(PTReducerMixIn, BatchMeanReducer):
    pass


class PTMeanPerChanelReducer(PTReducerMixIn, MeanPerChReducer):
    pass


def maybe_add_squeeze(aggregator, squeeze_dims):
    if not squeeze_dims:
        return aggregator

    def post_aggregation_hook(aggregated_value):
        return PTNNCFCollectorTensorProcessor.squeeze(PTNNCFTensor(aggregated_value), dim=squeeze_dims).tensor

    return PostAggregateHook(aggregator=aggregator, post_aggregation_hook=post_aggregation_hook)


def get_min_max_statistic_collector(
    use_abs_max,
    reducers_axes,
    reducers_keepdims: bool,
    aggregators_axes,
    aggregators_keepdims,
    num_samples: int,
    squeeze_dims,
):
    tensor_collector = TensorCollector(PTMinMaxTensorStatistic)

    aggregator_kwargs = {
        "tensor_processor": PTNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "aggregation_axes": aggregators_axes,
        "keepdims": aggregators_keepdims,
    }
    min_reducer = PTMinReducer(reducers_axes, keepdims=reducers_keepdims)
    min_aggregator = MinAggregator(**aggregator_kwargs)
    min_aggregator = maybe_add_squeeze(min_aggregator, squeeze_dims)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MIN_STAT, min_reducer, min_aggregator)

    max_reducer_cls = PTAbsMaxReducer if use_abs_max else PTMaxReducer
    max_reducer = max_reducer_cls(reducers_axes, keepdims=reducers_keepdims)
    max_aggregator = MaxAggregator(**aggregator_kwargs)
    max_aggregator = maybe_add_squeeze(max_aggregator, squeeze_dims)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MAX_STAT, max_reducer, max_aggregator)
    return tensor_collector


def get_mixed_min_max_statistic_collector(
    reducers_axes,
    reducers_keepdims: bool,
    aggregators_axes,
    aggregators_keepdims,
    use_abs_max: bool,
    use_means_of_mins: bool,
    use_means_of_maxs: bool,
    squeeze_dims,
    num_samples: int = None,
    window_size: int = None,
):
    tensor_collector = TensorCollector(PTMinMaxTensorStatistic)
    min_reducer = PTMinReducer(reducers_axes, keepdims=reducers_keepdims)

    kwargs = {
        "tensor_processor": PTNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "aggregation_axes": aggregators_axes,
        "keepdims": aggregators_keepdims,
        "window_size": window_size,
    }
    min_aggregator_cls = MeanAggregator if use_means_of_mins else MinAggregator
    min_aggregator = min_aggregator_cls(**kwargs)
    min_aggregator = maybe_add_squeeze(min_aggregator, squeeze_dims)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MIN_STAT, min_reducer, min_aggregator)

    max_reducer_cls = PTAbsMaxReducer if use_abs_max else PTMaxReducer
    max_reducer = max_reducer_cls(reducers_axes, keepdims=reducers_keepdims)
    max_aggregator_cls = MeanAggregator if use_means_of_maxs else MinAggregator
    max_aggregator = max_aggregator_cls(**kwargs)
    max_aggregator = maybe_add_squeeze(max_aggregator, squeeze_dims)
    tensor_collector.register_statistic_branch(PTMinMaxTensorStatistic.MAX_STAT, max_reducer, max_aggregator)

    return tensor_collector


def get_median_mad_statistic_collector(
    reducers_axes,
    reducers_keepdims: bool,
    aggregators_axes,
    aggregators_keepdims,
    num_samples: int,
    squeeze_dims,
    window_size: int = None,
):
    return _get_collection_without_reduction(
        MedianAbsoluteDeviationAggregator,
        PTMedianMADTensorStatistic,
        reducers_axes=reducers_axes,
        reducers_keepdims=reducers_keepdims,
        aggregators_axes=aggregators_axes,
        aggregators_keepdims=aggregators_keepdims,
        num_samples=num_samples,
        squeeze_dims=squeeze_dims,
        window_size=window_size,
    )


def get_precentile_tensor_collector(
    percentiles_to_collect,
    reducers_axes,
    reducers_keepdims: bool,
    aggregators_axes,
    aggregators_keepdims,
    num_samples: int,
    squeeze_dims,
    window_size: int = None,
):
    return _get_collection_without_reduction(
        partial(PrecentileAggregator, percentiles_to_collect=percentiles_to_collect),
        PTPercentileTensorStatistic,
        reducers_axes=reducers_axes,
        reducers_keepdims=reducers_keepdims,
        aggregators_axes=aggregators_axes,
        aggregators_keepdims=aggregators_keepdims,
        num_samples=num_samples,
        squeeze_dims=squeeze_dims,
        window_size=window_size,
    )


def _get_collection_without_reduction(
    aggregator_cls,
    statistic_class,
    reducers_axes,
    reducers_keepdims: bool,
    aggregators_axes,
    aggregators_keepdims,
    num_samples: int,
    squeeze_dims,
    window_size: int = None,
):
    tensor_collector = TensorCollector(statistic_class)
    reducer = PTNoopReducer()
    aggregation_axes = list(set(list(aggregators_axes) + [dim + 1 for dim in reducers_axes]))
    aggregator = aggregator_cls(
        PTNNCFCollectorTensorProcessor,
        aggregation_axes=aggregation_axes,
        window_size=window_size,
        num_samples=num_samples,
        keepdims=True,
    )
    dims_to_squeeze = [0] if squeeze_dims else []
    dims_to_squeeze += [axis + 1 for axis in reducers_axes] if not reducers_keepdims else []
    dims_to_squeeze += aggregators_axes if not aggregators_keepdims else []
    if dims_to_squeeze:

        def post_aggregation_hook(aggregated_value):
            retval = {}
            for key, value in aggregated_value.items():
                retval[key] = PTNNCFCollectorTensorProcessor.squeeze(PTNNCFTensor(value), dim=dims_to_squeeze).tensor
            return retval

        aggregator = PostAggregateHook(aggregator=aggregator, post_aggregation_hook=post_aggregation_hook)

    tensor_collector.register_statistic_branch(
        PTMedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY, reducer, aggregator
    )
    return tensor_collector


def get_mean_percentile_statistic_collector(
    percentiles_to_collect,
    reducers_axes,
    reducers_keepdims: bool,
    aggregators_axes,
    aggregators_keepdims,
    num_samples: int,
    squeeze_dims,
    window_size: int = None,
):
    tensor_collector = TensorCollector(PTPercentileTensorStatistic)
    quantiles_to_collect = np.true_divide(percentiles_to_collect, 100)
    reducer = PTQuantileReducer(reduction_axes=reducers_axes, quantile=quantiles_to_collect, keepdims=reducers_keepdims)
    for output_port_id, p in enumerate(percentiles_to_collect):
        aggregator = MeanAggregator(
            PTNNCFCollectorTensorProcessor,
            aggregation_axes=aggregators_axes,
            keepdims=aggregators_keepdims,
            num_samples=num_samples,
            window_size=window_size,
        )
        aggregator = maybe_add_squeeze(aggregator, squeeze_dims)
        tensor_collector.register_statistic_branch(
            (PTPercentileTensorStatistic.PRECENTILE_VS_VALUE_DICT, p), reducer, aggregator, output_port_id
        )
    return tensor_collector


def get_mean_stat_collector(num_samples, channel_axis, window_size=None):
    if channel_axis == 0:
        reducer = PTBatchMeanReducer()
    else:
        reducer = PTMeanPerChanelReducer(channel_axis)
    noop_reducer = PTNoopReducer()

    kwargs = {
        "tensor_processor": PTNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "window_size": window_size,
    }
    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(PTMeanTensorStatistic)
    collector.register_statistic_branch(PTMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(PTMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


PT_REDUCERS_MAP = {
    StatisticsType.MIN: PTMinReducer,
    StatisticsType.MAX: PTMaxReducer,
    StatisticsType.ABS_MAX: PTAbsMaxReducer,
    StatisticsType.MEAN: PTMeanReducer,
    StatisticsType.QUANTILE: PTQuantileReducer,
    StatisticsType.ABS_QUANTILE: PTAbsQuantileReducer,
}
