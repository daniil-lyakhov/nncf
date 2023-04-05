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

from abc import ABC
from abc import abstractmethod
from collections import deque
from collections import defaultdict
from typing import TypeVar, Tuple, Optional, List, Set, Dict, Any, Union

from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor import TensorType


InplaceInsertionFNType = TypeVar('InplaceInsertionFNType')


class TensorReducerBase(ABC):
    """
    Tensor reducer is a callable object that reduces given tensor according to
    the specified rule. Could handle tensors inplace or out of place.
    """

    def __init__(self,
                 reduction_shape: Optional[ReductionShape] = None,
                 inplace: bool = False):
        """
        :param reduction_shape: Reduction shape for reduction calculation. Equal to list(range(len(input.shape)))
            if empty.
        :param: Wheather should be calculated inplace or out of place.

        """
        self._reduction_shape = reduction_shape
        self._tensor_processor: NNCFCollectorTensorProcessor = self._get_processor()
        self._inplace = inplace

    @property
    def inplace(self):
        return self._inplace

    @property
    def output_port_id(self) -> int:
        return 0

    @classmethod
    def name(cls):
        return cls.__name__

    @staticmethod
    @abstractmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        pass

    @abstractmethod
    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        """
        Specifies the reduction rule in terms of NNCFCollectorTensorProcessor.

        :param x: Tensor to register.
        """

    @abstractmethod
    def get_output_name(self, target_node_name: str,
                        port_id: int) -> str:
        """
        Returns target output name from target model that is
            modified for statistic collection.

        :param target_node_name: Target node name for reducer.
        :param port_id: Target port id for target node name for reducer.
        :return: Target output name for reducer.
        """

    @abstractmethod
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        """
        Returns correspondent inplace operation builder if inplace operations are available in backend.

        :return: Inplace operation builder if possible else None.
        """

    def __call__(self, x: TensorType):
        if self.inplace:
            return x

        if self._reduction_shape is None:
            self._reduction_shape = tuple(range(len(x.shape)))
        return self._reduce_out_of_place(x)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and\
            self._reduction_shape == __o._reduction_shape and\
            self._inplace == __o.inplace

    def __hash__(self) -> int:
        return hash((self.name(), self._inplace))


class TensorAggregatorBase:
    """
    Tensor aggregator is designed to recieve (register) calculated statistics and
    aggregate them in terms of NNCFCollectorTensorProcessor operations.
    """

    def __init__(self, tensor_processor: NNCFCollectorTensorProcessor,
                 num_samples: Optional[int]):
        """
        :param tensor_processor: Backend-specific tensor processor.
        :param num_samples: Maximum number of samples to collect. Aggregator
            skips tensor registration if tensor registration was called num_samples times before.
            Aggregator never skips registration if num_samples is None.
        """

        self._tensor_processor = tensor_processor
        self._num_samples = num_samples
        self._collected_samples = 0
        self._container = []

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @classmethod
    def name(cls):
        return cls.__name__

    def register_reduced_input(self, x: TensorType):
        if self._num_samples is not None and \
            self._collected_samples >= self._num_samples:
            return
        self._register_reduced_input_impl(x)
        self._collected_samples += 1

    @abstractmethod
    def _register_reduced_input_impl(self, x: TensorType) -> None:
        """
        Registers incoming tensor in tensor aggregator.

        :param x: Tensor to register.
        """

    @abstractmethod
    def aggregate(self) -> Any:
        """
        Aggregates collected tensors and returns aggregated result.

        :retunr: Aggregated result.
        """

    def reset(self):
        self._collected_samples = 0
        self._container = []

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and \
            self._num_samples == __o.num_samples

    def __hash__(self) -> int:
        return hash((self.name()))


class TensorCollector:
    """
    Calculates statistics at given tensors acording to registered statistic branches.
    Statistic branch consist of one reducer and one aggregator instance. TensorCollector
    applies reducer on a correspondent input and then passes reduced tensor to
    a correspondent aggregator for each registered statistic branch.
    Recieves tesnors by `register_input` method. Aggregated values as a TensorStatistic instance or
    a dict could be collected by `get_statistics` call.
    """

    def __init__(self,
                 statistic_container: Optional[TensorStatistic] = None
                 ) -> None:
        self._reducers: Set[TensorReducerBase] = set()
        self._aggregators: Dict[Tuple[int, int], TensorAggregatorBase] = {}
        self._stat_container_kwargs_map: Dict[str, Tuple[int, int]] = {}
        self._stat_container = statistic_container
        self._enabled = True

    @property
    def num_samples(self) -> int:
        return max(aggregator.num_samples for aggregator in self._aggregators.values())

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reducers(self):
        return self._reducers.copy()

    @property
    def aggregators(self):
        return self._aggregators.copy()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def register_statistic_branch(self, container_key: str,
                                  reducer: TensorReducerBase, aggregator: TensorAggregatorBase) -> None:
        """
        Registers statistic collection branch for a container key. Correspondent input will be reduced
        by given reducer and reduced value will be registered and aggregated by given aggregator.
        Passed container key should be unique for the TensorCollector instance.
        Passed aggregator instance should never be used twice for one TensorCollector instance.

        :param container_key: Container key to pass aggregated statistic to.
        :param reducer: TensorReducer instance for the statistic collection branch.
        :param aggregator: TensorAggergator instance for the statistic collection branch.
        """
        if container_key in self._stat_container_kwargs_map:
            raise RuntimeError(f'Two differend statistic branches for one'
                               f' container key {container_key} are encountered')
        if any(aggr is aggregator for aggr in self._aggregators.values()):
            raise RuntimeError(f'One aggregator instance {aggregator} '
                               f' for different branches is encountered')

        self._reducers.add(reducer)
        key = (hash(reducer), hash(aggregator))

        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        self._stat_container_kwargs_map[container_key] = key

    def get_output_info(self, target_node_name: str, port_id: int) -> List[Tuple[int, str]]:
        """
        Returns list of pairs of reducers names and correspondent output names.

        :param target_node_name: Target node name to assemble output name.
        :param port_id: Target node specific port id to assemble output name.
        :returns: List of pairs of reducers hashes and correspondent output names.
        """
        retval = []
        for reducer in self._reducers:
            retval.append((hash(reducer), reducer.get_output_name(target_node_name, port_id)))
        return retval

    def register_inputs(self, inputs: Dict[str, TensorType]) -> None:
        """
        Registers given input in TensorCollector.

        :param inputs: Tensor inputs in format of dict where keys
            are reducer names and values are correspondent input tensors
        """
        if not self._enabled:
            return

        reduced_inputs = {}
        for reducer in self._reducers:
            reducer_hash = hash(reducer)
            input_ = inputs[reducer_hash]
            reduced_inputs[reducer_hash] = reducer(input_)

        for (reducer_hash, _), aggregator, in self._aggregators.items():
            aggregator.register_reduced_input(reduced_inputs[reducer_hash])

    def _aggregate(self) -> None:
        result = {}
        for key, aggregator, in self._aggregators.items():
            val = aggregator.aggregate()
            result[key] = val
        return result

    def get_statistics(self) -> Union[TensorStatistic, Dict[str, Any]]:
        """
        Returns aggregated values in format of a TensorStatistic instance or
        a dict.

        :returns: Aggregated values.
        """

        aggregated_values = self._aggregate()
        kwargs = {}
        for container_key, branch_key in self._stat_container_kwargs_map.items():
            kwargs[container_key] = aggregated_values[branch_key]

        if not self._stat_container:
            return kwargs
        return self._stat_container(**kwargs)

    def get_inplace_fn_info(self) -> List[Tuple[Any, int]]:
        """
        Returns necessery information to insert inplace operation into graph.

        :returns: nesessery information to insert inplace operation into graph
            in format of pair of reducer builder and correspondent reducer output port id.
        """
        retval = []
        for reducer in self._reducers:
            if reducer.inplace:
                retval.append((reducer.get_inplace_fn(), reducer.output_port_id))
        return retval

    def any_stat_out_of_place(self) -> bool:
        """
        Returns True if any reducer is calculated out of place.

        :returns: True if any reducer is calculated out of place.
        """
        return any(not reducer.inplace for reducer in self._reducers)

    def replace_aggregator(self, key: Tuple[int, int], aggregator: TensorAggregatorBase) -> None:
        """
        Friend method that replaces aggregator instance on equivalent one.
        Key shoud be valid for for given aggregator and a statistic branch
        with key should be present in TensorCollector.

        :param key: Statistic branch key.
        :param aggregator: Aggregator instance to replace existing instance by given key.
        """
        assert key in self._aggregators
        assert key[1] == hash(aggregator)
        self._aggregators[key] = aggregator

    def reset(self):
        for aggregator in self._aggregators.values():
            aggregator.reset()


class MergedTensorCollector(TensorCollector):
    """
    Tensor collector that merge several tensor collectors in one.
    Statistics collected by a merged tensor collector automatically available
    in all tensor collectors that were merged by the merged tensor collector.
    This works because merged tensor collectors share tensor aggregators instances with
    the merged tensor collector.
    """

    def __init__(self, tensor_collectors: List[TensorCollector]) -> None:
        """
        :param tensor_collectors: Tensor collectors to merge.
        """
        super().__init__()
        aggregators: Dict[Tuple[int, int], List[Tuple[TensorCollector, TensorAggregatorBase]]] =\
            defaultdict(list)
        for tensor_collector in tensor_collectors:
            if not tensor_collector.enabled:
                continue
            self._reducers.update(tensor_collector.reducers)
            for key, aggregator in tensor_collector.aggregators.items():
                aggregators[key].append((tensor_collector, aggregator))

        for key, aggregators_to_merge in aggregators.items():
            _, unique_aggregator = aggregators_to_merge[0]
            for tensor_collector, _ in aggregators_to_merge[1:]:
                tensor_collector.replace_aggregator(key, unique_aggregator)
            self._aggregators[key] = unique_aggregator


##################################################Reducers##################################################


class NoopReducer(TensorReducerBase):
    def __init__(self):
        super().__init__(inplace=False)

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return None

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        return x


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        return self._tensor_processor.reduce_min(x, self._reduction_shape)


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        return self._tensor_processor.reduce_max(x, self._reduction_shape)


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        x = self._tensor_processor.abs(x)
        return self._tensor_processor.reduce_max(x, self._reduction_shape)

class BatchMeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        return self._tensor_processor.batch_mean(x)


class MeanPerChReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        return self._tensor_processor.mean_per_channel(x, self._reduction_shape)


class NumpyConverter(TensorReducerBase):
    def __init__(self):
        super().__init__(None, False)

    def _reduce_out_of_place(self, x: TensorType) -> TensorType:
        return self._tensor_processor.to_numpy(x)


##################################################Aggregators##################################################


class NoopAggregator(TensorAggregatorBase):
    def __init__(self, num_samples: Optional[int]):
        super().__init__(None, num_samples)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container.append(x.tensor)

    def aggregate(self):
        return self._container


class OnlineMinAggregator(TensorAggregatorBase):
    def _register_reduced_input_impl(self, x: TensorType) -> None:
        if not self._container:
            self._container = x
        else:
            self._container = self._tensor_processor.min(x, self._container)

    def aggregate(self):
        return self._container.tensor


class OnlineMaxAggregator(TensorAggregatorBase):
    def _register_reduced_input_impl(self, x: TensorType) -> None:
        if not self._container:
            self._container = x
        else:
            self._container = self._tensor_processor.max(x, self._container)

    def aggregate(self):
        return self._container.tensor


class ShapeAggregator(TensorAggregatorBase):
    def __init__(self):
        super().__init__(None, 1)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container = x

    def aggregate(self):
        return self._container.tensor.shape


class OfflineMinMaxAggregatorBase(TensorAggregatorBase):
    def __init__(self, tensor_processor, use_per_sample_stats: bool,
                 num_samples: Optional[int], window_size=None):
        super().__init__(tensor_processor, num_samples)
        self._window_size = window_size
        self._container = deque(maxlen=window_size)
        self._use_per_sample_stats = use_per_sample_stats

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        if self._use_per_sample_stats:
            self._container.extend(self._tensor_processor.unstack(x))
        else:
            self._container.append(x)


class OfflineMinAggregator(OfflineMinMaxAggregatorBase):
    def aggregate(self):
        stacked_val = self._tensor_processor.stack(self._container)
        return self._tensor_processor.reduce_min(stacked_val, axis=0).tensor


class OfflineMaxAggregator(OfflineMinMaxAggregatorBase):
    def aggregate(self):
        stacked_val = self._tensor_processor.stack(self._container)
        return self._tensor_processor.reduce_max(stacked_val, axis=0).tensor


class OfflineMeanAggregator(OfflineMinMaxAggregatorBase):
    def aggregate(self):
        stacked_val = self._tensor_processor.stack(self._container)
        return self._tensor_processor.mean(stacked_val, axis=0).tensor