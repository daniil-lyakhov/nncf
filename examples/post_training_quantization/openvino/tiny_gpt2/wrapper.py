import numpy as np
import openvino.runtime as ov
from collections import defaultdict
from typing import Any


class NNCFOVWrappedModel:
    def __init__(self, ov_model, custom_forward, set_ov_model, **kwargs) -> None:
        self._ov_model = ov_model
        self._original_model_outputs_names = {op.node.friendly_name for op in ov_model.outputs}
        self._custom_forward = custom_forward
        self._set_ov_model = set_ov_model
        self._collected_statistics = defaultdict(list)
        self._stack_axis = 0
        self._ov_statistics_model = None
        self._kwargs = kwargs

    def __getattr__(self, __name: str) -> Any:
        return object.__getattribute__(self._ov_model, __name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self._ov_statistics_model is None:
            raise RuntimeError()
        return self._custom_forward(self, self._ov_statistics_model, *args, **kwds)

    def set_statistics_ov_model(self, ov_model):
        self._ov_statistics_model = ov.Core().compile_model(ov_model, device_name="CPU")

    @property
    def collected_statistics(self):
        aggregated_statistics = {}
        for friendly_name, values in self._collected_statistics.items():
            aggregated_statistics[friendly_name] = np.stack(values, axis=self._stack_axis)
        return aggregated_statistics

    def collect_statistics_callback(self, *args):
        # Take all not original outputs and save to self._collected_statistics
        if len(args) == 1:
            outputs = args[0]
            assert isinstance(outputs, dict)
        else:
            assert len(args) == 2
            outputs = {k: v for k, v in zip(*args)}
        original_model_output = {}
        for op, value in outputs.items():
            if op.node.friendly_name in self._original_model_outputs_names:
                original_model_output[op] = value
                continue
            if not isinstance(value, np.ndarray):
                value = value.data
            self._collected_statistics[op.node.friendly_name].append(value)
        if len(args) == 1:
            return original_model_output
        return zip(*[(k, v) for k, v in original_model_output.items()])
