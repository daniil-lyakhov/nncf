"""
 Copyright (c) 2021 Intel Corporation
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

from typing import List

import tensorflow as tf

from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper


def apply_saturation_issue_fix(model: tf.keras.Model, op_names: List[str]) -> tf.keras.Model:
    if not isinstance(model, tf.keras.Model):
        raise ValueError(
            'Expected model to be a `tf.keras.Model` instance but got: {}'.format(type(model)))

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            for weight_attr, ops in layer.weights_attr_ops.items():
                for op_name in ops:
                    if op_name in op_names:
                        apply_saturation_fix_to_layer(layer, weight_attr, op_name)


def apply_saturation_fix_to_layer(layer: NNCFWrapper, weight_attr: str, op_name: str) -> None:
    layer_weight = layer.layer_weights[weight_attr]
    op = layer.weights_attr_ops[weight_attr][op_name]
    ops_weights = layer.ops_weights[op_name]
    layer_weight.assign(
        op.call(layer_weight, ops_weights, False)
    )
    layer.set_layer_weight(weight_attr, layer_weight)
    op.apply_saturation_fix(ops_weights)
