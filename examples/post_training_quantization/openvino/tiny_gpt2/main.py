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

def _contiguous_helper(tensor: np.ndarray) -> np.ndarray:
    return tensor if tensor.flags["C_CONTIGUOUS"] else np.ascontiguousarray(tensor)


def custom_forward(self, model, dataitem):
    hf_model = self._kwargs['hf_model']
    hf_model.forward = _custom_forward(hf_model, self)
    hf_model.model = model
    hf_model.request = model.create_infer_request()
    hf_model.generate(
        **dataitem, min_length=GENERATION_LENGTH, max_length=GENERATION_LENGTH, num_beams=1
    )
    return self.collected_statistics


def _custom_forward(self, wrapper):
    def _custom_forward_wrapped(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        self.compile()

        if self.use_cache and past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs = {}
        if past_key_values is not None:
            # Flatten the past_key_values
            past_key_values = tuple(
                _contiguous_helper(np.array(past_key_value))
                for pkv_per_layer in past_key_values
                for past_key_value in pkv_per_layer
            )
            # Add the past_key_values to the decoder inputs
            inputs = {
                input_name: Tensor(past_key_value, shared_memory=True)
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values)
            }

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = (
                self.normalized_config.num_attention_heads if self.config.model_type == "bloom" else 1
            )
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())

        inputs["input_ids"] = np.array(input_ids)

        # Add the attention_mask inputs when needed
        if "attention_mask" in self.input_names and attention_mask is not None:
            inputs["attention_mask"] = np.array(attention_mask)

        # Run inference
        self.request.start_async(inputs)
        self.request.wait()

        # !!!PATCH HERE!!!
        model_outputs, request_outputs = wrapper.collect_statistics_callback(self.request.model_outputs, self.request.outputs)
        outputs = {
            key.get_any_name(): value.data for key, value in zip(model_outputs, request_outputs)
        }
        logits = torch.from_numpy(outputs["logits"]).to(self.device)

        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(
                torch.from_numpy(outputs[key]).to(self.device) for key in self.key_value_output_names
            )
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
    return _custom_forward_wrapped

wrapped_model = NNCFOVWrappedModel(model_with_pkv.model, custom_forward, None, hf_model=model_with_pkv)

isinstance(wrapped_model, NNCFOVWrappedModel)

def transform_fn(data_item):
    return data_item

def infer_model(model, model_inputs, model_outputs):
    return model.generate(
        **model_inputs, min_length=GENERATION_LENGTH, max_length=GENERATION_LENGTH, num_beams=1
    )

dataset = nncf.CustomInferenceDataset([tokens] * 10, transform_fn, custom_forward)

# Check for user
output = None
data_item = next(iter(dataset.get_inference_data()))
output = infer_model(model_with_pkv, data_item, None)

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

quantized_model = nncf.quantize(wrapped_model, dataset, subset_size=3)
