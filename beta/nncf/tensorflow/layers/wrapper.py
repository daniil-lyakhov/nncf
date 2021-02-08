"""
 Copyright (c) 2020 Intel Corporation
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

from collections import OrderedDict
from inspect import getfullargspec

import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import _DictWrapper

from beta.nncf.tensorflow.layers.custom_objects import get_nncf_custom_objects
from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.operation import InputType


@NNCF_CUSTOM_OBJECTS.register()
class NNCFWrapper(tf.keras.layers.Wrapper):
    """
    This wrapper augments a keras layer so the NNCF Operations may be applied to weights,
    callable attributes (like activations), input and output of the wrapped layer.
    """
    def __init__(self, layer, **kwargs):
        """
        Create a pruning wrapper for a keras layer.

        :param layer: the keras layer to be wrapped
        :param kwargs: additional keyword arguments to be passed to the keras layer.
        """
        if layer is None:
            raise ValueError('`layer` cannot be None.')

        if not isinstance(layer, tf.keras.layers.Layer) or \
                isinstance(layer, tf.keras.Model):
            raise ValueError(
                '`layer` can only be a `tf.keras.layers.Layer` instance. '
                'You passed an instance of type: {input}.'.format(
                    input=layer.__class__.__name__))

        if 'name' not in kwargs:
            kwargs['name'] = '{}_{}'.format('nncf_wrapper', layer.name)

        super().__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')

        self.weights_attr_ops = {}
        #TODO: add
        # self.inputs_ops = OrderedDict()
        # self.outputs_ops = OrderedDict()
        # self.pre_callable_attr_ops = {}
        # self.post_callable_attr_ops = {}
        # self.pre_hook = {}
        # self.post_hook = {}

        # TODO: add
        # if not hasattr(self, '_batch_input_shape') and hasattr(
        #         layer, '_batch_input_shape'):
        #     self._batch_input_shape = self.layer._batch_input_shape

        self._init_layer_call_fn_args()
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._ops_weights = {}
        self._layer_weights = {}

    @property
    def trainable(self):
        return self.layer.trainable

    @trainable.setter
    def trainable(self, value):
        self.layer.trainable = value

    @property
    def trainable_weights(self):
        return self._trainable_weights + self.layer.trainable_weights \
                                        + self._get_ops_weights_by_condition(trainable=True)

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights + self.layer.non_trainable_weights \
                                            + self._get_ops_weights_by_condition(trainable=False)

    @property
    def updates(self):
        return self.layer.updates + self._updates

    @property
    def losses(self):
        return self.layer.losses + self._losses

    @property
    def data_format(self):
        return getattr(self.layer, 'data_format', 'channels_last')

    @property
    def ops_weights(self):
        return self._ops_weights

    @property
    def layer_weights(self):
        return self._layer_weights

    def build(self, input_shape=None):
        super().build(input_shape)
        for weight_attr, ops in self.weights_attr_ops.items():
            weight = self.get_layer_weight(weight_attr)
            for op_name, op in ops.items():
                self._ops_weights[op_name] = op.build(
                    weight.shape, InputType.WEIGHTS, weight_attr, self)
            self._layer_weights[weight_attr] = weight
            self._trainable_weights.append(weight)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        self._apply_ops(training)

        if self._expects_training_arg:
            outputs = self.layer.call(inputs, training=training)
        else:
            outputs = self.layer.call(inputs)

        return outputs

    def _get_ops_weights_by_condition(self, trainable=True):
        result = []
        for _, ops in self.weights_attr_ops.items():
            for op_name, op in ops.items():
                if op.trainable == trainable:
                    ops_weight = self._ops_weights[op_name]
                    # TODO: unify
                    if isinstance(ops_weight, _DictWrapper):
                        for weight in ops_weight.values():
                            result.append(weight)
                    else:
                        result.append(ops_weight)
        return result

    def _apply_ops(self, training):
        for weight_attr, ops in self.weights_attr_ops.items():
            layer_weight = self._layer_weights[weight_attr]
            for op_name, op in ops.items():
                layer_weight = op(layer_weight,
                                  self._ops_weights[op_name],
                                  training)
            self.set_layer_weight(weight_attr, layer_weight)

    def registry_weight_operation(self, weights_attr, op, op_name=None):
        if weights_attr not in self.weights_attr_ops:
            self.weights_attr_ops[weights_attr] = OrderedDict()

        if op_name is None:
            if op.name is None:
                op_name = 'nncf_op_{}:{}'.format(weights_attr, len(self.weights_attr_ops[weights_attr]))
            else:
                op_name = op.name

        self.weights_attr_ops[weights_attr][op_name] = op
        return op_name

    def get_op_by_name(self, name):
        '''Return op by name if exist
        else None'''
        for _, ops in self.weights_attr_ops.items():
            for op_name, op in ops.items():
                if op_name == name:
                    return op


    def get_operation_weights(self, operation_name):
        return self._ops_weights[operation_name]

    def get_layer_weight(self, weight_attr):
        weight = getattr(self.layer, weight_attr, None)
        if weight is not None:
            return weight
        for w in self.layer.weights:
            if w.name.split(":")[0] == weight_attr:
                return w
        return None

    def set_layer_weight(self, weight_attr, weights):
        if hasattr(self.layer, weight_attr):
            setattr(self.layer, weight_attr, weights)
        else:
            self._layer_weights[weight_attr].assign(weights)

    def _init_layer_call_fn_args(self):
        call_full_argspec = getfullargspec(self.layer.call)
        call_fn_args = self._get_call_fn_args(call_full_argspec)
        self._expects_training_arg = "training" in call_fn_args

    @staticmethod
    def _get_call_fn_args(call_full_argspec):
        all_args = call_full_argspec.args + call_full_argspec.kwonlyargs
        if all_args and all_args[0] == 'self':
            return all_args[1:]
        return all_args

    def get_config(self):
        config = super().get_config()

        weights_attr_ops = {}
        for weights_attr, ops in self.weights_attr_ops.items():
            weights_attr_ops[weights_attr] = []
            for op_name in ops:
                op_config = tf.keras.utils.serialize_keras_object(ops[op_name])
                op_config['name'] = op_name
                weights_attr_ops[weights_attr].append(op_config)
        config['weights_attr_operations'] = weights_attr_ops
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()

        weights_attr_ops_config = config.pop('weights_attr_operations')

        layer = tf.keras.layers.deserialize(config.pop('layer'), custom_objects=custom_objects)
        wrapper = cls(layer=layer, **config)

        for weights_attr, operations in weights_attr_ops_config.items():
            for op_config in operations:
                wrapper.registry_weight_operation(
                    weights_attr,
                    tf.keras.layers.deserialize(
                        op_config,
                        custom_objects=get_nncf_custom_objects()),
                    op_config.get('name', None)
                )

        return wrapper
