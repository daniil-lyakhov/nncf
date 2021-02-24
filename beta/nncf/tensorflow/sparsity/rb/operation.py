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

import tensorflow as tf

from beta.nncf.tensorflow.graph.utils import get_weight_by_name
from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.layers.operation import NNCFOperation
from beta.nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from beta.nncf.tensorflow.sparsity.rb.functions import calc_rb_binary_mask, st_binary_mask, binary_mask
from beta.nncf.tensorflow.functions import logit

OP_NAME = 'rb_sparsity_mask_apply'

@NNCF_CUSTOM_OBJECTS.register()
class RBSparsifyingWeight(NNCFOperation):

    def __init__(self, eps=1e-6):
        '''Setup trainable param'''
        super().__init__(name=OP_NAME,
                         trainable=True)
        self.eps = eps

    # TODO: make it static
    def build(self, input_shape, input_type, name, layer):
        if input_type is not InputType.WEIGHTS:
            raise ValueError(
                'RB Sparsity mask operation could not be applied to input of the layer: {}'.
                    format(layer.name))

        mask = layer.add_weight(
            name + '_mask',
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(logit(0.99)),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN)

        trainable = layer.add_weight(
            name + '_trainable',
            initializer=tf.keras.initializers.Constant(1),
            trainable=False,
            dtype=tf.int8)

        return {
            'mask': mask,
            'trainable': trainable,
        }

    # TODO: make it static
    def call(self, layer_weights, op_weights, _):
        '''Apply rb sparsity mask to given weights
        :param layer_weights: target weights to sparsify
        :param op_weights: operation weights contains
           mask and param trainable
        :param _:'''
        if tf.equal(op_weights['trainable'], tf.constant(1, dtype=tf.int8)):
            return apply_mask(layer_weights, calc_rb_binary_mask(op_weights['mask'], self.eps))
        return tf.stop_gradient(apply_mask(layer_weights, binary_mask(op_weights['mask'])))

    def freeze(self, op_weights):
        op_weights['trainable'].assign(0)
        self._trainable = False

    @staticmethod
    def loss(mask):
        '''Return count of non zero weight in mask'''
        return tf.cast(tf.reduce_sum(binary_mask(mask)), tf.int32)
