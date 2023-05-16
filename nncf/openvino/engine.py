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

from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import openvino.runtime as ov

from nncf.common.engine import Engine
from nncf.data import Sequence
from nncf.parameters import TargetDevice

SEQUENTIAL_SAMPLE_STACK_AXIS = 0


class OVNativeEngine(Engine):
    """
    Implementation of the engine for OpenVINO backend.

    OVNativeEngine uses
    [OpenVINO Runtime](https://docs.openvino.ai/latest/openvino_docs_OV_UG_OV_Runtime_User_Guide.html)
    to infer the model.
    """

    def __init__(self, model: ov.Model, target_device: TargetDevice = TargetDevice.CPU):
        if target_device == TargetDevice.ANY:
            target_device = TargetDevice.CPU

        ie = ov.Core()
        self.compiled_model = ie.compile_model(model, target_device.value)
        self.input_tensor_names = set()
        self.number_of_inputs = len(model.inputs)
        for model_input in model.inputs:
            self.input_tensor_names.update(model_input.get_names())

    def _check_input_data_format(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> None:
        """
        Checks correspondence of the model input names and the passed data.
        If there is a mismatch, the method throws a more specific and readable error than
        original error raised by the compiled model.

        :param input_data: Provided inputs to infer the model.
        """
        actual_num_inputs = 1 if isinstance(input_data, np.ndarray) else len(input_data)
        if actual_num_inputs != self.number_of_inputs:
            raise RuntimeError(f"Model expects {self.number_of_inputs} inputs, but {actual_num_inputs} are provided.")
        if isinstance(input_data, dict):
            for name in input_data:
                if isinstance(name, str) and name not in self.input_tensor_names:
                    raise RuntimeError(f"Missing a required input: {name} to run the model.")

    def infer(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if isinstance(input_data, Sequence):
            return self._sequential_infer(input_data)
        return self._infer(input_data)

    def _sequential_infer(self, sequence: Sequence):
        model_output = None
        model_outputs = defaultdict(list)
        for token in sequence.get_tokens_iter():
            filled_inputs = sequence.fill_inputs(token, model_output)
            model_output = self._infer(filled_inputs)
            for output_name, output_value in model_output.items():
                model_outputs[output_name].append(output_value)

        # Stack model outputs and return them
        stacked_outputs = {}
        for output_name, output_values in model_outputs.items():
            stacked_outputs[output_name] = np.stack(output_values, axis=SEQUENTIAL_SAMPLE_STACK_AXIS)

        return stacked_outputs

    def _infer(
        self, input_data: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray], Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Runs model on the provided input via OpenVINO Runtime.
        Returns the dictionary of model outputs by node names.

        :param input_data: Inputs for the model.
        :return output_data: Model's output.
        """
        self._check_input_data_format(input_data)
        model_outputs = self.compiled_model(input_data)

        output_data = {}
        for tensor, value in model_outputs.items():
            for tensor_name in tensor.get_names():
                output_data[tensor_name] = value
        return output_data
