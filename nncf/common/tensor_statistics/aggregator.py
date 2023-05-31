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
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from itertools import islice
from typing import Any, Dict, TypeVar

import numpy as np
from tqdm import tqdm

from nncf.common.factory import EngineFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.data.dataset import CustomInferenceDataset
from nncf.data.dataset import Dataset

TensorType = TypeVar("TensorType")
TModel = TypeVar("TModel")


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    STACK_AXIS = 0

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.stat_subset_size = None
        self.statistic_points = StatisticPointsContainer()
        self._is_custom_inference = isinstance(dataset, CustomInferenceDataset)

    def collect_statistics(self, model: TModel) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: backend-specific model instance
        """
        model_transformer = ModelTransformerFactory.create(model)

        merged_statistics = self._get_merged_statistic_points(self.statistic_points, model)
        if self._is_custom_inference:
            merged_statistics = self._adapt_collectors(merged_statistics, self.STACK_AXIS)

        transformation_layout = self._get_transformation_layout_extra_outputs(merged_statistics)
        model_with_outputs = model_transformer.transform(transformation_layout)
        engine = EngineFactory.create(model_with_outputs)
        if self._is_custom_inference:
            sequence_container = defaultdict(list)
            custom_forward = self.dataset.get_custom_forward(
                engine.compiled_model, self._get_callback(model, sequence_container)
            )

        for input_data in tqdm(
            islice(self.dataset.get_inference_data(), self.stat_subset_size),
            total=self.stat_subset_size,
            desc="Statistics collection",
        ):
            if self._is_custom_inference:
                custom_forward(input_data)
                processed_outputs = {}
                for friendly_name, values in sequence_container.items():
                    processed_outputs[friendly_name] = self._get_tensor_processor().stack(values, axis=self.STACK_AXIS)
            else:
                processed_outputs = engine.infer(input_data)
                processed_outputs = self._process_outputs(processed_outputs)
            self._register_statistics(processed_outputs, merged_statistics)

    @staticmethod
    def _get_callback(model, sequence_container: StatisticPointsContainer):
        pass

    def register_statistic_points(self, statistic_points: StatisticPointsContainer) -> None:
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.

        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """
        for _, _statistic_points in statistic_points.items():
            for _statistic_point in _statistic_points:
                self.statistic_points.add_statistic_point(_statistic_point)

        for _, _statistic_points in self.statistic_points.items():
            for _statistic_point in _statistic_points:
                for _, tensor_collectors in _statistic_point.algorithm_to_tensor_collectors.items():
                    for tensor_collector in tensor_collectors:
                        if self.stat_subset_size is None:
                            self.stat_subset_size = tensor_collector.num_samples
                        elif tensor_collector.num_samples is not None:
                            self.stat_subset_size = max(self.stat_subset_size, tensor_collector.num_samples)

    @abstractmethod
    def _register_statistics(self, outputs: Dict[str, NNCFTensor], statistic_points: StatisticPointsContainer) -> None:
        """
        Process prepared raw model outputs and statistic points for the further usage.

        :param outputs: prepared raw model outputs
        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """

    @abstractmethod
    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        """
        Creates backend-specific transformation layout for the further statistics collection.

        :param statistic_points: StatisticPointsContainer to add outputs
        :return: TransformationLayout with the corresponding transformations
        """

    @staticmethod
    @abstractmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel
    ) -> StatisticPointsContainer:
        """
        Creates a new StatisticPointContainer that has no duplicated tensor collectors for one
        unique statistic point. Alters statistic collectors in the given statistic point container so statistics
        collected by merged statistic collectors will be available in all corresponding statistic collectors
        from the given statistic point container.

        :param statistic_points: Registered statistic points with possible tensor collectors duplicates.
        :param model: Backend-specific target model.
        :return: Merged statistic points container bounded with given statistic point container.
        """

    @staticmethod
    def _adapt_collectors(statistic_points: StatisticPointsContainer, stack_axis: int):
        return statistic_points

    @staticmethod
    @abstractmethod
    def _process_outputs(outputs: Any) -> Dict[str, NNCFTensor]:
        """
        Post-process model outputs for the further statistics collection.

        :param outputs: raw model outputs
        :return: processed model outputs in Dict[str, NNCFTensor] format
        """

    @staticmethod
    @abstractmethod
    def _get_tensor_processor():
        pass
