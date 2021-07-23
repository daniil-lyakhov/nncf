import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.python.framework.convert_to_constants import _run_inline_graph_optimization


class ModelType:
    KerasLayer = 'KerasLayerModel'
    FuncModel = 'FuncModel'
    SubClassModel = 'SubclassModel'


def second_dummy_model():
    inputs = tf.keras.Input((1000, ))
    x = tf.keras.layers.Dense(10)(inputs)
    x = tf.keras.layers.Dense(1001)(x)
    return tf.keras.Model(inputs, x)


class SubclassModel(tf.keras.Model):
    def __init__(self):
        super(SubclassModel, self).__init__()
        self.first_submodule = tf.keras.applications.mobilenet_v2.MobileNetV2()
        self.second_submodule = second_dummy_model()

    def call(self, inputs, training=None, mask=None):
        x = self.first_submodule(inputs)
        x = self.second_submodule(x)
        return x


def get_func_model():
    mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3),
                                                               include_top=False)
    input = tf.keras.Input((224, 224, 3))
    x = mobilenet(input)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1001)(x)
    return (tf.keras.Model(input, x),)


def get_submoduled_model():
    return (SubclassModel(),)


def get_KerasLayer_model():
    """

    :return : 2 * (KerasLayer instance, non optimized concrete function,
               optimized graph_def) first one is trainable and second one is not
    """
    retval = []
    for trainable in [True, False]:
        #assert not tf.distribute.has_strategy(), 'Can\'t modify KerasLayer graph created in cross replica mode'

        keras_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5",
                                     trainable=trainable, arguments=dict(batch_norm_momentum=0.997))
        tf_f = tf.function(lambda x: keras_layer.call(x, training=trainable))
        concrete = tf_f.get_concrete_function(*[tf.TensorSpec((None, 224, 224, 3), tf.float32, name='input')])
        optimized_gd = _run_inline_graph_optimization(concrete, False, False)
        retval.append({
                'layer': keras_layer,
                'graph_def': optimized_gd,
                'concrete_function': concrete,
            })
    return retval


def get_model(model_type: str):
    if model_type == 'FuncModel':
        return get_func_model()
    if model_type == 'SubclassModel':
        return get_submoduled_model()
    if model_type == 'KerasLayerModel':
        return get_KerasLayer_model()

    raise ValueError('Wrong model type')
