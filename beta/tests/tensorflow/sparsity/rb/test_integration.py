import tensorflow as tf
import tensorflow_addons as tfa
import os

from beta.nncf import NNCFConfig
from beta.nncf.helpers.callback_creation import create_compression_callbacks
from beta.tests.tensorflow.helpers import create_compressed_model_and_algo_for_test


def get_basic_sparsity_config(model_size=4, input_sample_size=None,
                              sparsity_init=0.02, sparsity_target=0.5, sparsity_target_epoch=2,
                              sparsity_freeze_epoch=3):
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]

    config = NNCFConfig()
    config.update({
        "model": "basic_sparse_conv",
        "model_size": model_size,
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression":
            {
                "algorithm": "rb_sparsity",
                "params":
                    {
                        "schedule": "polynomial",
                        "sparsity_init": sparsity_init,
                        "sparsity_target": sparsity_target,
                        "sparsity_target_epoch": sparsity_target_epoch,
                        "sparsity_freeze_epoch": sparsity_freeze_epoch
                    },
            }
    })
    return config


def train_lenet():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.transpose(tf.reshape(x_train, (-1, 1, 28, 28)), (0, 2, 3, 1))
    x_test = tf.transpose(tf.reshape(x_test, (-1, 1, 28, 28)), (0, 2, 3, 1))

    x_train = x_train / 255
    x_test = x_test / 255

    inp = tf.keras.Input((28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 5)(inp)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(48, 5)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dense(84)(x)
    y = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=y)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(5e-4),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=64, epochs=16, validation_split=0.2,
              callbacks=tf.keras.callbacks.ReduceLROnPlateau())

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    model.save('LeNet.h5')


def test_rb_sparse_target_lenet():
    if not os.path.exists('LeNet.h5'):
        train_lenet()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.transpose(tf.reshape(x_train, (-1, 1, 28, 28)), (0, 2, 3, 1))
    x_test = tf.transpose(tf.reshape(x_test, (-1, 1, 28, 28)), (0, 2, 3, 1))

    x_train = x_train / 255
    x_test = x_test / 255

    model = tf.keras.models.load_model('LeNet.h5')

    config = get_basic_sparsity_config(sparsity_init=0.05, sparsity_target=0.3,
                                       sparsity_target_epoch=10, sparsity_freeze_epoch=15)
    compress_model, compress_algo = create_compressed_model_and_algo_for_test(model, config)
    compression_callbacks = create_compression_callbacks(compress_algo, log_tensorboard=True, log_dir='logdir/')

    class SparsityRateTestCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            target = compress_algo.loss.target_sparsity_rate
            actual = compress_algo.raw_statistics()['sparsity_rate_for_sparsified_modules']
            print(f'target {target}, actual {actual}')
            assert abs(actual - target) < 0.09

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

    metrics = [loss_obj,
               tfa.metrics.MeanMetricWrapper(compress_algo.loss,
                                             name='rb_loss')]

    compress_model.add_loss(compress_algo.loss)

    compress_model.compile(
        loss=loss_obj,
        optimizer=tf.keras.optimizers.Adam(5e-3),
        metrics=metrics,
    )

    compress_model.fit(x_train, y_train, batch_size=64, epochs=15, validation_split=0.2,
                       callbacks=[tf.keras.callbacks.ReduceLROnPlateau(),
                                  compression_callbacks,
                                  SparsityRateTestCallback()])

    test_scores = compress_model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    compress_model.save('LeNet_rb_sparse.h5')
