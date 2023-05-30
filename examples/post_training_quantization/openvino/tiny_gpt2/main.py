import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from openvino.runtime import Core, Tensor
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast

from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from optimum.utils import NormalizedConfigManager


from openvino.tools import mo
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM
from examples.post_training_quantization.openvino.tiny_gpt2.wrapper import NNCFOVWrappedModel
import nncf


GENERATION_LENGTH = 20


model_id = "hf-internal-testing/tiny-random-gpt2"
#model_id = "hf-internal-testing/tiny-random-GPTNeoModel"
#model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokens = tokenizer("This is a sample input", return_tensors="pt")

model_with_pkv = OVModelForCausalLM.from_pretrained(model_id, export=True, use_cache=True)


def set_ov_model_in_hf_model(hf_model, ov_model):
    hf_model.model = ov_model
    hf_model.request = ov_model.create_infer_request()


def get_custom_forward(ov_model, callback_fn):
    hf_model = model_with_pkv
    set_ov_model_in_hf_model(hf_model, ov_model)

    def _callback_fn(info):
        outputs = {k: v for k, v in zip(info["infer_request"].model_outputs, info["infer_request"].outputs)}
        callback_fn(outputs)
    hf_model.request.set_callback(_callback_fn, {"infer_request": hf_model.request})

    def custom_forward(dataitem):
        hf_model.generate(
            **dataitem, min_length=GENERATION_LENGTH, max_length=GENERATION_LENGTH, num_beams=1
        )
    return custom_forward


def transform_fn(data_item):
    return data_item


dataset = nncf.CustomInferenceDataset([tokens] * 10, transform_fn, get_custom_forward)


# Fix ov model duplicated names:
names = set()
for op in model_with_pkv.model.get_ops():
    friendly_name = op.get_friendly_name()
    while True:
        if friendly_name not in names:
            break
        friendly_name += "_"
    names.add(friendly_name)
    op.set_friendly_name(friendly_name)

quantized_model = quantized_model = nncf.quantize(model_with_pkv.model, dataset, subset_size=3)

model_with_pkv.model = quantized_model
model_with_pkv.request = None
