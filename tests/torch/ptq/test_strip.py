# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy

import pytest
import torch
from torch.quantization import FakeQuantize

import nncf
from nncf.torch import load_from_config
from nncf.torch.strip_tuned_lora_model import strip_tuned_lora_model
from tests.torch.helpers import BasicConvTestModel
from tests.torch.ptq.test_weights_compression import ShortTransformer
from tests.torch.test_compressed_graph import check_graph


@pytest.mark.parametrize("strip_type", ("nncf", "torch", "nncf_interfere"))
@pytest.mark.parametrize("do_copy", (True, False), ids=["copy", "inplace"])
def test_nncf_strip_api(strip_type, do_copy):
    model = BasicConvTestModel()
    quantized_model = nncf.quantize(model, nncf.Dataset([torch.ones(model.INPUT_SIZE)]), subset_size=1)
    quantized_model.nncf.get_graph().visualize_graph("fq_model.dot")
    if strip_type == "nncf":
        strip_model = nncf.strip(quantized_model, do_copy)
    elif strip_type == "torch":
        strip_model = nncf.torch.strip(quantized_model, do_copy)
    elif strip_type == "nncf_interfere":
        strip_model = quantized_model.nncf.strip(do_copy)

    if do_copy:
        assert id(strip_model) != id(quantized_model)
    else:
        assert id(strip_model) == id(quantized_model)

    for fq in strip_model.nncf.external_quantizers.values():
        assert isinstance(fq, FakeQuantize)


def test_strip_lora_adapters(_seed):
    model = ShortTransformer(64, 16)
    model.wte.weight = torch.nn.Parameter(0.1 * torch.rand((16, 64)))
    model.linear.weight = torch.nn.Parameter(0.1 * torch.rand((64, 64)))
    model.lm_head.weight = torch.nn.Parameter(0.1 * torch.rand((16, 64)))
    op_name = "quantizer_op_name"
    nncf_config = {
        "compression_state": [
            {
                "type": "PTSharedFnInsertionCommand",
                "target_points": [
                    {
                        "target_type": {"name": "OPERATOR_PRE_HOOK"},
                        "input_port_id": 1,
                        "target_node_name": "ShortTransformer/Linear[linear]/linear_0",
                    }
                ],
                "op_name": op_name,
                "compression_module_type": 0,
                "compression_module_name": "AsymmetricQuantizer",
                "fn_config": {
                    "num_bits": 4,
                    "mode": "asymmetric",
                    "signedness_to_force": None,
                    "narrow_range": False,
                    "half_range": False,
                    "scale_shape": [64, 1, 1],
                    "weight_shape": [64, 64],
                    "device": "cpu",
                    "logarithm_scale": False,
                    "is_quantized_on_export": False,
                    "compression_lr_multiplier": None,
                },
                "hooks_group_name": "default_hooks_group",
                "priority": 0,
            }
        ],
        "trace_parameters": True,
    }
    ex_input = torch.ones((16,), dtype=torch.int)
    nncf_model = load_from_config(model, nncf_config, ex_input)
    nncf_model.nncf.get_graph().visualize_graph("nncf_graph.dot")
    nncf_model = nncf_model.to(torch.float16)
    with torch.no_grad():
        quantizer = nncf_model.nncf.external_quantizers[op_name]
        quantizer.input_range *= 0.35
        quantizer._lora_A -= 0.9
        quantizer._lora_B += 0.01

        stripped_nncf_model = strip_tuned_lora_model(deepcopy(nncf_model))
        orig_output = nncf_model(ex_input)
        striped_output = stripped_nncf_model(ex_input)
        diff = (orig_output - striped_output).abs()
        assert (diff < torch.finfo(torch.float16).eps).all()

    nncf_graph = stripped_nncf_model.nncf.get_graph()
    check_graph(nncf_graph, "stripped_lora_adapters_model.dot", "strip", extended=True)
