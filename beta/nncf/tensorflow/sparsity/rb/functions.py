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

from beta.nncf.tensorflow.functions import logit, st_threshold


def binary_mask(mask):
    return tf.round(tf.math.sigmoid(mask))


def st_binary_mask(mask):
    return st_threshold(tf.math.sigmoid(mask))


def calc_rb_binary_mask(mask, generators, eps=0.01):
    # TODO: check in distributed mode (mirrored strategy)
    # TODO: remove pylint disable comment
    # when https://github.com/tensorflow/tensorflow/pull/46046 will be merged into the release
    def to_args(gs):
        def f():
            return [gs[tf.distribute.get_replica_context().replica_id_in_sync_group]]
        return tf.distribute.get_strategy().run(f)
    args = to_args(generators)
    def f(g):
        return g.uniform(mask.shape, minval=0, maxval=1)
    uniform = tf.distribute.get_strategy().run(f, args=args)
    mask = mask + logit(tf.clip_by_value(uniform, eps, 1 - eps))
    return st_binary_mask(mask)
