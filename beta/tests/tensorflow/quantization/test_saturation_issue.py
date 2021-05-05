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

import pytest
import numpy as np
import tensorflow as tf

from beta.tests.tensorflow.helpers import get_basic_two_conv_test_model
from beta.tests.tensorflow.helpers import get_basic_n_conv_test_model
from beta.tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from nncf.common.quantization.structs import QuantizationMode
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from beta.nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from beta.nncf.tensorflow.quantization.quantizers import QuantizerConfig
from beta.nncf.tensorflow.quantization.quantizers import Quantizer
from beta.tests.tensorflow.quantization.utils import get_basic_quantization_config
from beta.tests.tensorflow.quantization.utils import get_basic_asym_quantization_config


DIM_SPLIT = 1000
EPS = 1e-6
TEST_MODELS = {
    'TwoConvTestModel': lambda: get_basic_two_conv_test_model(
        input_shape=(4, 4, 1),
        out_channels=2,
        kernel_size=2,
        weight_init=-1.,
        bias_init=-2.,
        transpose=False),
    'TwoConvTestModelTranspose': lambda: get_basic_two_conv_test_model(
        input_shape=(4, 4, 1),
        out_channels=1,
        kernel_size=2,
        weight_init=-1.,
        bias_init=-2.,
        transpose=True),
    'EightConvTestModel': lambda: get_basic_n_conv_test_model(
        input_shape=(24, 24, 1),
        in_out_ch=((1, 3), (3, 5), (5, 7), (7, 10)),
        kernel_sizes=(2,) * 4,
        weight_init=-1.,
        bias_init=-2.,
        transpose=False)
}


def check_quantized_values_equals(y_train, y_val, eps, range_len, narrow_range):
    diff = np.abs(y_val - y_train)
    if np.max(diff) > eps:
        # If any point gets in really close to the middle of quant
        # it can changes its quant due to rounding error
        outlayers = diff[diff > 1e-6]
        quant_len = range_len / (128 - (2 if narrow_range else 1))
        assert np.max(np.abs(outlayers - quant_len)) < eps


def check_quantized_values_clipped_correctly(y_val, low, range_len, eps, narrow_range):
    if low > eps:
        # Range greater than zero
        assert np.min(y_val) > -EPS
        assert np.max(y_val) < range_len + EPS
    elif low + range_len < eps:
        # Range lower than zero
        assert np.min(y_val) > -range_len - EPS
        assert np.max(y_val) < EPS
    else:
        # Range with zero
        min_adj = Quantizer._min_adj(7., low, range_len, narrow_range)# pylint: disable=protected-access
        assert np.min(y_val) > min_adj - eps
        assert np.max(y_val) < min_adj + range_len + eps


@pytest.mark.parametrize('bits,low,range_,narrow_range,ref',
                         [(7, -1., 2., False, -128/127),
                          (7, -2., 2., True, -2.)], ids=['full_range', 'narrow_range'])
def test_min_adj(bits, low, range_, narrow_range, ref):
    res = Quantizer._min_adj(bits, low, range_, narrow_range).numpy()# pylint: disable=protected-access
    assert abs(res - ref) < EPS


@pytest.mark.parametrize('per_ch', [False, True], ids=['per_tensor', 'per_channel'])
@pytest.mark.parametrize('signedness_to_force', [True, False], ids=['signed', 'unsigned'])
@pytest.mark.parametrize('narrow_range', [False, True], ids=['full_range', 'narrow_range'])
def test_symmetric_quantize_equal_on_train_and_val(per_ch, signedness_to_force, narrow_range):
    qconfig = QuantizerConfig(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=signedness_to_force,
        per_channel=per_ch)
    qspec = TFQuantizerSpec.from_config(
        qconfig,
        narrow_range=narrow_range,
        half_range=True)

    quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qspec.mode)
    quantizer = quantizer_cls('quantizer', qspec)
    weights = quantizer.build((DIM_SPLIT,), InputType.INPUTS, 'dummy_name', tf.keras.layers.Layer())

    ref_signed_var = -1. if signedness_to_force else 0.
    ref_scale = 1.
    assert (weights['scale_var'].numpy() == ref_scale).all()
    assert (weights['signed_var'].numpy() == ref_signed_var).all()

    x = tf.Variable(np.linspace(-3, 3, DIM_SPLIT), dtype=tf.float32)
    y_train = quantizer(x, weights, training=True).numpy()
    y_val = quantizer(x, weights, training=False).numpy()

    range_len = 2. if signedness_to_force else 1.
    check_quantized_values_equals(y_train, y_val, EPS, range_len, narrow_range)
    check_quantized_values_clipped_correctly(y_val, ref_scale * ref_signed_var, range_len, EPS, narrow_range)


@pytest.mark.parametrize('low,range_len', [(-1., 2.), (-5., 4.), (3., 2.)],
                         ids=['zero_in_range', 'max_l_than_zero', 'low_g_than_zero'])
@pytest.mark.parametrize('narrow_range', [False, True], ids=['full_range', 'narrow_range'])
def test_asymmetric_quantize_equal_on_train_and_val_per_tensor(low, range_len, narrow_range):
    qconfig = QuantizerConfig(
        num_bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        per_channel=False)
    qspec = TFQuantizerSpec.from_config(
        qconfig,
        narrow_range=narrow_range,
        half_range=True)

    quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qspec.mode)
    quantizer = quantizer_cls('quantizer', qspec)
    weights = quantizer.build((DIM_SPLIT,), InputType.WEIGHTS, 'dummy_name', tf.keras.layers.Layer())

    weights['input_low_var'].assign(low)
    weights['input_range_var'].assign(range_len)
    x = tf.constant(np.linspace(-5, 5, DIM_SPLIT), dtype=tf.float32)

    y_train = quantizer(x, weights, training=True).numpy()
    y_val = quantizer(x, weights, training=False).numpy()
    check_quantized_values_equals(y_train, y_val, EPS, range_len, narrow_range)
    check_quantized_values_clipped_correctly(y_val, low, range_len, EPS, narrow_range)


@pytest.mark.parametrize('shape', [(3 * DIM_SPLIT,), (1, 3 * DIM_SPLIT), (1, 1, 1, 3 * DIM_SPLIT)])
@pytest.mark.parametrize('narrow_range', [False, True], ids=['full_range', 'narrow_range'])
def test_assym_quantize_equal_on_train_and_val_per_ch(shape, narrow_range):
    qconfig = QuantizerConfig(
        num_bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=None,
        per_channel=True)
    qspec = TFQuantizerSpec.from_config(
        qconfig,
        narrow_range=narrow_range,
        half_range=True)

    quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qspec.mode)
    quantizer = quantizer_cls('quantizer', qspec)
    weights = quantizer.build(shape, InputType.WEIGHTS, 'dummy_name', tf.keras.layers.Layer())

    # weights[:DIM_SPLIT] is range [-1, 2] with zero point inside
    # weights[DIM_SPLIT:2*DIM_SPLIT] is range [-5, -2] with all values < 0
    # weights[2*DIM_SPLIT:] is range [3, 5] with all values > 0
    range_len = 3
    low = tf.repeat(tf.constant([-1, -5, 3], dtype=tf.float32), repeats=[DIM_SPLIT] * 3)
    range_ = tf.repeat(tf.constant([range_len] * 3, dtype=tf.float32), repeats=[DIM_SPLIT] * 3)

    weights['input_low_var'].assign(low)
    weights['input_range_var'].assign(range_)

    x = tf.reshape(tf.concat([tf.constant(np.linspace(-5, 5, DIM_SPLIT), dtype=tf.float32)] * 3, axis=0), shape)
    y_train = quantizer(x, weights, training=True).numpy().flatten()
    y_val = quantizer(x, weights, training=False).numpy().flatten()
    check_quantized_values_equals(y_train, y_val, EPS, range_len, narrow_range)

    # Check values clipped correctly
    # Range with zero
    check_quantized_values_clipped_correctly(y_val[:DIM_SPLIT], -1., range_len, EPS, narrow_range)
    # Range lower than zero
    check_quantized_values_clipped_correctly(y_val[DIM_SPLIT:2*DIM_SPLIT], -5., range_len, EPS, narrow_range)
    # Range greater than zero
    check_quantized_values_clipped_correctly(y_val[2*DIM_SPLIT:], 3., range_len, EPS, narrow_range)


@pytest.mark.parametrize('model_cls', list(TEST_MODELS.values()), ids=list(TEST_MODELS.keys()))
@pytest.mark.parametrize('quantize_mode', ['sym', 'asym'],
                         ids=['sym', 'asym'])
def test_quants_at_train_are_the_same_as_at_val(model_cls, quantize_mode):
    config = get_basic_quantization_config() if quantize_mode == 'sym' else \
        get_basic_asym_quantization_config()
    model, _ = create_compressed_model_and_algo_for_test(model_cls(), config)
    dummy_x = tf.random.uniform((5,) + model.input_shape[1:])
    y_on_train = model(dummy_x, training=True)
    y_on_val = model(dummy_x, training=False)
    tf.debugging.assert_near(y_on_train, y_on_val)
