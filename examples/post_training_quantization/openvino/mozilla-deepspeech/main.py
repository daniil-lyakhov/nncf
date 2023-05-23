import os
import subprocess
from typing import Dict

import numpy as np
import openvino.runtime as ov
from openvino.tools.accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
from openvino.tools.pot.configs.config import Config

import nncf

model_name = "mozilla-deepspeech-0.6.1"
cache_dir = os.path.dirname(__file__)
dataset_config = os.path.join(cache_dir, "accuracy_checker.json")

command = f"omz_downloader --name {model_name} --cache_dir {cache_dir}"
cmd_output = subprocess.call(command, shell=True)  # nosec

model_dir = os.path.join(cache_dir, model_name)
if not os.path.exists(model_dir):
    command = f"omz_converter --name {model_name} -o {os.path.join(cache_dir, model_name)}"
    cmd_output = subprocess.call(command, shell=True)  # nosec

xml_path = os.path.join(model_dir, f"public/{model_name}/FP16/{model_name}.xml")
ov_model = ov.Core().read_model(xml_path)

config = Config.read_config(dataset_config)
config.configure_params()
accuracy_checker_config = config.engine

model_evaluator = create_model_evaluator(accuracy_checker_config)
model_evaluator.load_network([{"model": ov_model}])
model_evaluator.select_dataset("")


def sequence_transform_fn(data_item):
    """
    Quantization transform function. Extracts and preprocesses sequential inputs data from dataloader
    for quantization, returns iterable on preprocessed elements of feeded data item.

    :param data_item:  Data item produced by DataLoader during iteration
    :return: Iterable object on preprocessed elements of feeded data item.
    """
    _, batch_annotation, batch_input, _ = data_item
    filled_inputs, _, _ = model_evaluator._get_batch_input(batch_input, batch_annotation)
    for filled_input in filled_inputs:
        input_data = {}
        for name, value in filled_input.items():
            input_data[model_evaluator.launcher.input_to_tensor_name[name]] = value
        yield input_data


def create_model_input_fn(model_inputs, model_outputs):
    """
    Combines preprocessed model inputs from `get_tokens_from_sequence_fn` and model outputs
    from previous iteration. None is feeded as model outputs on first iteration.

    :param model_inputs: Preprocessed model input from `get_token_from_sequence_fn`.
    :param model_outputs: Outuputs of target model from previous iteration. None on first iteration.
    :return: Dict of acutual model inputs combined from preprocessed model input from `get_token_from_sequence_fn`
        and previous model outputs for sequential models.
    """
    state_inputs = model_evaluator.launcher._fill_lstm_inputs(model_outputs)
    model_inputs.update(state_inputs)
    return model_inputs


dataset = nncf.CustomInferenceDataset(model_evaluator.dataset, sequence_transform_fn, create_model_input_fn)

# Check for user
output = None
data_item = next(dataset.get_inference_data())
sequence = sequence_transform_fn(data_item)
for sequence_item in sequence:
    input = create_model_input_fn(sequence_item, output)
    output = ov_model(input)

quantized_model = nncf.quantize(ov_model, dataset, subset_size=3)