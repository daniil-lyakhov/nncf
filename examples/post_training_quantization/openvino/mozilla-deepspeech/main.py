import json
import os
import subprocess

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


def get_tokens_from_sequence_func(data_item):
    _, batch_annotation, batch_input, _ = data_item
    filled_inputs, _, _ = model_evaluator._get_batch_input(batch_input, batch_annotation)
    for filled_input in filled_inputs:
        input_data = {}
        for name, value in filled_input.items():
            input_data[model_evaluator.launcher.input_to_tensor_name[name]] = value
        yield input_data


def fill_sequential_inputs_fn(model_inputs, model_outputs):
    # Combine model inputs with state model outputs
    # or fill state model outputs if model_outputs is None
    state_inputs = model_evaluator.launcher._fill_lstm_inputs(model_outputs)
    model_inputs.update(state_inputs)
    return model_inputs


dataset = nncf.RecurentDataset(model_evaluator.dataset, get_tokens_from_sequence_func, fill_sequential_inputs_fn)
quantized_model = nncf.quantize(ov_model, dataset, subset_size=3)
