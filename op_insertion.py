import tensorflow as tf
import numpy as np

from typing import List
from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
from tensorflow.python.distribute.values_util import get_current_replica_id_as_int
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from itertools import zip_longest

from contrib import input_to_ops
from examples.tensorflow.classification.test_models import ModelType


DUMP_GRAPH = False


class InsertionPoint(object):
    WEIGHTS = 'w'
    AFTER_LAYER = 'after'
    BEFORE_LAYER = 'before'


class QuantizationSetup(object):
    def __init__(self, signed=None, should_init=None):
        self.signed = signed
        self.init_value = should_init


class NNCFCallableGraph(object):
    pass


class NNCFWrapperCustom(tf.keras.layers.Wrapper):
    def __init__(self, trainable_model, eval_model=None, **kwargs):
        super().__init__(tf.keras.layers.Layer(), **kwargs)
        self.model_type = ModelType.FuncModel
        self.trainable_model = NNCFCallableGraph()
        self.eval_model = NNCFCallableGraph()
        self.mirrored_vars_created = False
        self.ops_vars_created = False
        self.initial_model_weights = None
        if isinstance(trainable_model, dict):
            self.model_type = ModelType.KerasLayer

            self.trainable_model.graph_def = trainable_model['graph_def']
            self.trainable_model.concrete = trainable_model['concrete_function']
            self.eval_model.graph_def = eval_model['graph_def']
            self.eval_model.concrete = eval_model['concrete_function']
        else:
            self.trainable_model.orig_model = trainable_model
            self.eval_model.orig_model = trainable_model

    def get_keras_layer_mobilenet_v2_fq_placing_simular_to_nncf2_0(self, g):
        """Hardcode fq placing for examples.classification.test_models.get_KerasLayer_model"""
        #Get blocks fq
        add_ops = [op for op in g.get_operations() if 'addv2' in op.type.lower()]
        assert len(add_ops) == 10
        depthwise_conv =\
            [op for op in g.get_operations() if 'expanded' in op.name.lower() and 'conv' in op.type.lower() and 'depthwise' in op.name.lower()]
        project_ops =\
            [op for op in g.get_operations() if 'expanded' in op.name.lower() and 'conv' in op.type.lower() and 'project' in op.name.lower()]
        expand_ops =\
            [op for op in g.get_operations() if 'expanded' in op.name.lower() and 'conv' in op.type.lower() and 'expand/' in op.name.lower()]
        assert len(depthwise_conv) == len(project_ops) == len(expand_ops) + 1

        depthwise_conv_relu = self.get_left_childs(g, depthwise_conv, 2, 'Relu6')
        expand_ops_relu = self.get_left_childs(g, expand_ops, 2, 'Relu6')
        project_bn = self.get_left_childs(g, project_ops, 1, 'FusedBatchNormV3')
        # First conv
        first_conv = [op for op in g.get_operations() if 'MobilenetV2/Conv/Conv2D' in op.name and 'conv' in op.type.lower()][0]
        first_conv_relu = self.get_left_childs(g, [first_conv], 2, 'Relu6')[0]
        # Tail
        last_conv = [op for op in g.get_operations() if 'MobilenetV2/Conv_1/Conv2D' in op.name and 'conv' in op.type.lower()][0]
        last_conv_relu = self.get_left_childs(g, [last_conv], 2, 'Relu6')[0]
        avg_pool = self.get_left_childs(g, [last_conv], 4, 'AvgPool')[0]
        prediction_mul = self.get_left_childs(g, [last_conv], 6, ['Conv2D', 'Mul'])[0]
        #
        # Create transformation
        #
        transformations = []
        # Transformations for blocks
        transformations.extend([(op, InsertionPoint.WEIGHTS, QuantizationSetup(signed=True)) for op in depthwise_conv])
        transformations.extend([(op, InsertionPoint.WEIGHTS, QuantizationSetup(signed=True)) for op in project_ops])
        transformations.extend([(op, InsertionPoint.WEIGHTS, QuantizationSetup(signed=True)) for op in expand_ops])

        transformations.extend([(op, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=False)) for op in depthwise_conv_relu])
        transformations.extend([(op, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=True)) for op in project_bn])
        transformations.extend([(op, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=False)) for op in expand_ops_relu])
        transformations.extend([(op, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=True)) for op in add_ops])
        # Transformations for first conv
        # FQ on inputs
        transformations.append((first_conv, InsertionPoint.BEFORE_LAYER, QuantizationSetup(signed=True)))
        # FQ on first conv weights
        transformations.append((first_conv, InsertionPoint.WEIGHTS, QuantizationSetup(signed=True)))
        # FQ after first conv relu
        transformations.append((first_conv_relu, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=False)))
        # Transformation for net tail
        transformations.append((last_conv, InsertionPoint.WEIGHTS, QuantizationSetup(signed=True)))
        transformations.append((last_conv_relu, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=False)))
        transformations.append((avg_pool, InsertionPoint.AFTER_LAYER, QuantizationSetup(signed=False)))
        transformations.append((prediction_mul, InsertionPoint.WEIGHTS, QuantizationSetup(signed=True)))
        assert len(transformations) == 117

        return transformations

    def build(self, input_shape=None):
        for training, model in zip([True, False], [self.trainable_model, self.eval_model]):
            if self.model_type != ModelType.KerasLayer:
                tf_f = tf.function(lambda x: model.orig_model.call(x, training=training))
                concrete = tf_f.get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)])

                sorted_vars = get_sorted_on_captured_vars(concrete)
                model.mirrored_variables = model.orig_model.variables

            else:
                concrete = make_new_func(model.graph_def,
                                         model.concrete.graph.captures,
                                         model.concrete.variables,
                                         model.concrete.inputs,
                                         model.concrete.outputs)

                sorted_vars = get_sorted_on_captured_vars(concrete)
                model.mirrored_variables = self.create_mirrored_variables(sorted_vars)

            if not self.initial_model_weights:
                self.initial_model_weights = self.get_numpy_weights_list(sorted_vars)

            # Save mapping for concrete per replica inputs
            model.bn_weights_names = set(['/'.join(v.name.split('/')[:-1]) for v in concrete.variables if 'replica' in v.name.lower()])
            model.sorted_concrete_vars_names = [v.name for v in sorted_vars]
            if model.bn_weights_names:
                mirrored_vars_extended = []
                for v_concrete_name in model.sorted_concrete_vars_names:
                    name, _ = name_without_replica_idx(v_concrete_name)
                    mirrored_vars_extended.extend([v for v in model.mirrored_variables
                                                   if name_without_replica_idx(v.name)[0] == name])

                model.mirrored_variables = mirrored_vars_extended

            # Add new op to layer
            if not self.ops_vars_created:
                self.op_vars = []
            enable_quantization = True
            if enable_quantization:
                new_vars = []
                with concrete.graph.as_default() as g:
                    transformations = self.get_keras_layer_mobilenet_v2_fq_placing_simular_to_nncf2_0(g)
                    # Insert given transformations
                    for op, insertion_point, setup in transformations:
                        def fq_creation(input_tensor, name, init_value=6):
                            return create_fq_with_weights(input_tensor=input_tensor,
                                                          name=name,
                                                          signed=setup.signed,
                                                          init_value=init_value)

                        if insertion_point == InsertionPoint.AFTER_LAYER:
                            new_vars.append(insert_op_after(g, op, 0, fq_creation, op.name))
                        elif insertion_point == InsertionPoint.BEFORE_LAYER:
                            new_vars.append(insert_op_before(g, op, 0, fq_creation, f'{op.name}_before_layer'))
                        elif insertion_point == InsertionPoint.WEIGHTS:
                            min_val, max_val = self.get_min_max_op_weights(g, op, concrete.inputs,
                                                                           self.initial_model_weights)
                            scale = max(abs(min_val), abs(max_val))
                            fq_creation_initialized = lambda input_tensor, name: fq_creation(input_tensor, name, scale)
                            new_vars.append(insert_op_before(g, op, 1, fq_creation_initialized, op.name))
                        else:
                            raise RuntimeError('Wrong insertion point in quantization algo')

                    model.output_tensor = g.outputs[0]

                if not self.ops_vars_created:
                    self.op_vars = new_vars
                    self.ops_vars_created = True
                    print(f'{len(transformations)} quantizers were added successfully')

                # Make new concrete to update captured_inputs.
                # This is needed for correct export.

                # Update captures
                if isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy):
                    new_ops_vars = get_zero_replica_from_mirrored_vars(self.op_vars)
                else:
                    new_ops_vars = self.op_vars
                old_captures = [(data, placeholder) for data, placeholder in concrete.graph.captures]
                new_captures = old_captures[:-len(self.op_vars)]

                for new_var, (_, placeholder) in zip(new_ops_vars, old_captures[-len(self.op_vars):]):
                    new_captures.append((new_var.handle, placeholder))
                new_variables = [v for v in concrete.variables] + new_ops_vars
                concrete = make_new_func(concrete.graph.as_graph_def(),
                                         new_captures,
                                         new_variables,
                                         concrete.inputs,
                                         concrete.outputs)

            else:
                model.output_tensor = concrete.graph.outputs[0]

            model.fn_train = concrete

        if DUMP_GRAPH:
            tf.io.write_graph(concrete.graph, '/tmp', 'mobilenetv2_sub_with_conv.pb')

    def call(self, inputs, training=None):
        model_obj = self.trainable_model if training else self.eval_model
        if isinstance(tf.distribute.get_strategy(), tf.distribute.MirroredStrategy):
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                # Map correspondent replica of MirroredVariable to replica concrete function
                replica_id = get_current_replica_id_as_int()
                new_variables = []
                new_captured = []
                for concrete_var_name, var, input_tensor in zip_longest(
                                                                model_obj.sorted_concrete_vars_names,
                                                                model_obj.mirrored_variables + self.op_vars,
                                                                model_obj.fn_train.inputs[1:]):
                    if concrete_var_name:
                        # Check if some variables from other replicas are needed for
                        # concrete function call
                        name, idx = name_without_replica_idx(concrete_var_name)
                        if name not in model_obj.bn_weights_names:
                            idx = replica_id

                    new_variables.append(var._get_replica(idx))
                    new_captured.append((var._get_replica(idx).handle, input_tensor))
            else:
                # On compile time don't change vars
                new_variables = model_obj.fn_train.graph.variables
                new_captured = model_obj.fn_train.graph.captures
        else: # not distributed mode
            new_variables = []
            new_captured = []
            for var, input_tensor in zip(model_obj.mirrored_variables + self.op_vars,
                                         model_obj.fn_train.inputs[1:]):
                new_variables.append(var)
                new_captured.append((var.handle, input_tensor))

        fn_train = make_new_func(model_obj.fn_train.graph.as_graph_def(),
                                 new_captured,
                                 new_variables,
                                 model_obj.fn_train.inputs,
                                 [model_obj.output_tensor])

        return fn_train(inputs)

    def get_min_max_op_weights(self, graph, op, placeholders, np_vars):
        try:
            placeholder = self.get_op_weights_placeholder(graph, op)
        except IndexError:
            print(f'CANT MAP {op.name}')
            return -6, 6

        placeholders_names = [p.name.split(':')[0] for p in placeholders[1:]]
        idx = placeholders_names.index(placeholder.name)
        weight = np_vars[idx][0]
        print(f'map {op.name} -----> {np_vars[idx][1]}')
        return np.min(weight), np.max(weight)

    def get_left_childs(self, graph, ops: List, depth: int, op_type: str = None):
        """Get child for each op given by ops list in given depth"""
        retval = []
        for op in ops:
            i = 0
            child = [op]
            while i < depth and len(child):
                child = OperationUtils.get_children_ops(graph, child[0])
                i += 1

            child = child[0]
            if op_type is not None:
                if isinstance(op_type, list):
                    assert child.type in op_type
                else:
                    assert child.type == op_type

            retval.append(child)

        return retval

    @staticmethod
    def get_op_weights_placeholder(graph, op):
        placeholder = op
        while placeholder.type != 'Placeholder':
            placeholder = OperationUtils.get_parent_ops(graph, placeholder)[-1]

        return placeholder

    def create_mirrored_variables(self, vars):
        if not self.mirrored_vars_created:
            retval = []
            for var in vars:
                mirrored_var = tf.Variable(var.numpy(),
                                           trainable=var.trainable,
                                           dtype=var.dtype,
                                           name=var.name.split(':')[0] + '_mirrored')
                retval.append(mirrored_var)
            self.mirrored_vars_created = True
            self.mirrored_vars_cache = retval
        else:
            retval = self.mirrored_vars_cache

        return retval

    @staticmethod
    def get_numpy_weights_list(vars):
        #retval = {}
        #for var in vars:
        #    retval[var.name] = var.numpy()

        #return retval
        return [(var.numpy(), var.name) for var in vars]


def get_zero_replica_from_mirrored_vars(vars):
    return [v._get_replica(0) for v in vars]


def name_without_replica_idx(name):
    name = name.split(':')[0]
    if 'replica' in name:
        idx = int(name.split('_')[-1])
        name = '/'.join(name.split('/')[:-1])
    else:
        idx = 0
    return name, idx


def insert_softmax_in_graph(fn_train):
    with fn_train.graph.as_default() as g:
        softmax = tf.nn.softmax(g.outputs[0])

        return make_new_func(g.as_graph_def(),
                             g.captures,
                             g.variables,
                             fn_train.inputs,
                             [softmax])


# Copyed from:tensorflow.contrib.quantize.python.common.DropStringPrefix tags/v1.15.0
def RerouteTensor(t0, t1, can_modify=None):
    """Reroute the end of the tensor t0 to the ends of the tensor t1.

    Args:
      t0: a tf.Tensor.
      t1: a tf.Tensor.
      can_modify: iterable of operations which can be modified. Any operation
        outside within_ops will be left untouched by this function.

    Returns:
      The number of individual modifications made by the function.
    """
    nb_update_inputs = 0
    consumers = t1.consumers()
    if can_modify is not None:
        consumers = [c for c in consumers if c in can_modify]
    consumers_indices = {}
    for c in consumers:
        consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
    for c in consumers:
        for i in consumers_indices[c]:
            c._update_input(i, t0)  # pylint: disable=protected-access
            nb_update_inputs += 1
    return nb_update_inputs


# Copied from pocketflow:learners.uniform_quantization_tf.utils.insert_quant_op
def insert_op_before(graph, target_op, input_idx, node_creation_fn, name):
    """Insert quantization operations before node on input_idx.

    Args:
    * graph: TensorFlow graph
    * node_name: activation node's name
    :return: count of fq inserted into model
    """
    target_parent = None
    output_idx = None
    target_op_parents = OperationUtils.get_parent_ops(graph, target_op)
    target_parent_output = target_op.inputs[input_idx]
    for op in target_op_parents:
        for i, outputs in enumerate(op.outputs):
            if outputs.name == target_parent_output.name:
                target_parent = op
                output_idx = i

    if target_parent is None or output_idx is None:
        raise RuntimeError(f'Can\'t find node parent, node name: {target_op.name}')

    # re-route the graph to insert quantization operations
    return insert_op_after(graph, target_parent, output_idx, node_creation_fn, name)


def insert_op_after(graph, target_parent, output_index, node_creation_fn, name):
    input_to_ops_map = input_to_ops.InputToOps(graph)
    consumer_ops = input_to_ops_map.ConsumerOperations(target_parent)
    insert_op_output_tensor, node_weights = node_creation_fn(target_parent.outputs[output_index], name)
    RerouteTensor(insert_op_output_tensor, target_parent.outputs[output_index], consumer_ops)
    return node_weights


def create_fq_with_weights(input_tensor, name, signed, init_value):
    """Should be called in graph context"""
    with variable_scope.variable_scope('new_node'):
        scale = variable_scope.get_variable(
            f'scale_{name}',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(init_value),#init_ops.constant_initializer(1),
            trainable=True)

        min = -scale if signed else 0.
        output_tensor = tf.quantization.fake_quant_with_min_max_vars(input_tensor, min, scale)
    return output_tensor, scale


def get_sorted_on_captured_vars(concrete_fun):
    sorted_vars = []
    for value_tensor, graph_name in concrete_fun.graph.captures:
        for layer_var in concrete_fun.variables:
            if layer_var.handle is value_tensor:
                sorted_vars.append(layer_var)
    return sorted_vars


def make_new_func(output_graph_def, captures, variables, inputs, outputs):
    new_input_names = [tensor.name for tensor in inputs]
    inputs_map = {
        tensor.name: tensor for tensor in inputs
    }
    new_output_names = [tensor.name for tensor in outputs]
    new_func = my_function_from_graph_def(output_graph_def,
                                          new_input_names,
                                          new_output_names,
                                          captures,)
    for input in new_func.inputs:
        input.set_shape(inputs_map[input.name].shape)
        break

    new_func.graph.variables = variables
    return new_func


def my_function_from_graph_def(graph_def, inputs, outputs, ref_captures):
    def _imports_graph_def():
        importer.import_graph_def(graph_def, name="")

    wrapped_import = wrap_function.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    wrapped_import.graph.reset_captures([(tensor, import_graph.get_tensor_by_name(placeholder.name))
                                         for tensor, placeholder in ref_captures])
    return wrapped_import.prune(
        nest.map_structure(import_graph.as_graph_element, inputs),
        nest.map_structure(import_graph.as_graph_element, outputs))


class OperationUtils:
    @staticmethod
    def get_parent_ops(graph, target_op):
        retval = {}
        target_op_inputs = [x.name for x in target_op.inputs]
        for op in graph.get_operations():
            for idx, i in enumerate(target_op_inputs):
                if i in [x.name for x in op.outputs]:
                    retval[idx] = op
            if len(retval) == len(target_op.inputs):
                break
        return [retval[i] for i in range(len(retval))]

    @staticmethod
    def get_children_ops(graph, target_op):
        retval = {}
        target_op_outputs = [x.name for x in target_op.outputs]
        for op in graph.get_operations():
            for idx, out in enumerate(target_op_outputs):
                if out in [x.name for x in op.inputs]:
                    retval[idx] = op
            if len(retval) == len(target_op.outputs):
                break
        return [retval[i] for i in range(len(retval))]
