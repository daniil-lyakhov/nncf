import numpy as np
import pytest

from functools import partial

import tests.common.pruning.dummy_types as dummy_types

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.pruning.export_helpers import DefaultMetaOp
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm


@pytest.mark.parametrize('dummy_op_class,accept_pruned_input', [(dummy_types.DummyOpInput, False),
                                                                (dummy_types.DummyOpOutput, True),
                                                                (dummy_types.DummyOpStopMaskForward, False)])
def test_stop_propagate_ops(dummy_op_class, accept_pruned_input):
    node = NNCFNode(0, 'dummy_node')
    assert dummy_op_class.accept_pruned_input(node) == accept_pruned_input
    dummy_op_class.mask_propagation(node, None)
    assert node.data['output_mask'] is None


@pytest.mark.parametrize('dummy_op_class', [dummy_types.DummyOpIdentityMaskForward, dummy_types.DummyOpBatchNorm])
def test_identity_mask_propogation_prune_ops(dummy_op_class):
    assert dummy_op_class.accept_pruned_input(None)
    graph = NNCFGraph()
    conv_op = graph.add_nncf_node('conv_op', 'conv', dummy_types.DummyConvMetatype)
    identity_ops = []
    for alias in dummy_op_class.get_all_op_aliases():
        identity_op = graph.add_nncf_node('identity', alias, dummy_types.DymmyIdentityMaskForwardMetatype)
        graph.add_edge_between_nncf_nodes(
            from_node_id=conv_op.node_id,
            to_node_id=identity_op.node_id,
            tensor_shape=[10] * 4,
            input_port_id=0,
            output_port_id=0,
            dtype=Dtype.FLOAT)
        identity_ops.append(identity_op)
    # Check with and without masks
    for output_mask in [None, np.ones((10,))]:
        conv_op = graph.get_node_by_id(conv_op.node_id)
        conv_op.data['output_mask'] = output_mask
        MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
        for identity_op in identity_ops:
            identity_op = graph.get_node_by_id(identity_op.node_id)
            assert np.all(identity_op.data['output_mask'] == output_mask)


@pytest.mark.parametrize('valid_masks', [None, True, False])
def test_elementwise_prune_ops(valid_masks):
    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node('conv_op_0', dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype)
    conv_op_1 = graph.add_nncf_node('conv_op_1', dummy_types.DummyConvMetatype.name, dummy_types.DummyConvMetatype)
    elementwise_op = graph.add_nncf_node('elementwise', dummy_types.DummyElementwiseMetatype.name,
                                         dummy_types.DummyElementwiseMetatype)
    add_node = partial(graph.add_edge_between_nncf_nodes,
                       tensor_shape=[10] * 4,
                       input_port_id=0,
                       output_port_id=0,
                       dtype=Dtype.FLOAT)
    # conv_op_0 -> elementwise
    add_node(from_node_id=conv_op_0.node_id,
             to_node_id=elementwise_op.node_id)

    # conv_op_1 -> elementwise
    add_node(from_node_id=conv_op_1.node_id,
             to_node_id=elementwise_op.node_id)

    masks = [np.ones((10,)), np.ones((10,))] if valid_masks is not None else None

    def set_masks(masks, ops):
        for conv_op, mask in zip(ops, masks):
            conv_op = graph.get_node_by_id(conv_op.node_id)
            conv_op.data['output_mask'] = mask

    if valid_masks is None or valid_masks:
        if valid_masks:
            set_masks(masks, [conv_op_0, conv_op_1])
        MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
        elementwise_op = graph.get_node_by_id(elementwise_op.node_id)
        assert np.all(elementwise_op.data['output_mask'] == masks)
    else:
        def check_wrong_masks(masks):
            with pytest.raises(AssertionError):
                set_masks(masks, [conv_op_0, conv_op_1])
                MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()

        masks[0][0] = 0
        check_wrong_masks(masks)
        masks[0] = np.concatenate([masks[1], np.array([1])])
        check_wrong_masks(masks)


@pytest.mark.parametrize('num_channels,num_groups,accept_pruned_input_ref', [(10, 10, True),
                                                                             (10, 5, False),
                                                                             (10, 1, False)])
def test_group_norm_pruning_ops(num_channels, num_groups, accept_pruned_input_ref):
    graph = NNCFGraph()
    conv_op = graph.add_nncf_node('conv_op', 'conv', dummy_types.DummyConvMetatype)
    group_norm_layer_attributes = GroupNormLayerAttributes(True, num_channels=num_channels,
                                                           num_groups=num_groups)
    group_norm_op = graph.add_nncf_node('identity', dummy_types.DummyGroupNormMetatype.name,
                                        dummy_types.DummyGroupNormMetatype,
                                        layer_attributes=group_norm_layer_attributes)
    assert dummy_types.DummyOpGroupNorm.accept_pruned_input(group_norm_op) == accept_pruned_input_ref
    graph.add_edge_between_nncf_nodes(
        from_node_id=conv_op.node_id,
        to_node_id=group_norm_op.node_id,
        tensor_shape=[10] * 4,
        input_port_id=0,
        output_port_id=0,
        dtype=Dtype.FLOAT)
    # Check with and without masks
    for output_mask in [None, np.ones((10,))]:
        conv_op = graph.get_node_by_id(conv_op.node_id)
        conv_op.data['output_mask'] = output_mask
        MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
        identity_op = graph.get_node_by_id(group_norm_op.node_id)
        assert np.all(identity_op.data['output_mask'] == output_mask)


class DummyMaskProducerMetatype(dummy_types.DummyDefaultMetatype):
    name = 'mask_producer'


@dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES.register(DummyMaskProducerMetatype.name)
class MockOpMaskProducer(DefaultMetaOp):
    additional_types = [DummyMaskProducerMetatype.name]
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph):
        pass


@pytest.mark.parametrize('transpose', [True, False], ids=['transpose', 'not_transpose'])
@pytest.mark.parametrize('layer_attributes,ref_accept_pruned_input,conv_type', [
    ({'in_channels': 5,
      'out_channels': 10,
      'groups': 1}, True, 'usual_conv'),
    ({'in_channels': 10,
      'out_channels': 20,
      'groups': 5}, False, 'grouped_conv_no_depthwise'),
    ({'in_channels': 10,
      'out_channels': 20,
      'groups': 10}, True, 'depthwise_conv')
],
    ids=['usual_conv',
         'grouped_conv_no_depthwise',
         'depthwise_conv']
)
def test_conv_pruning_ops(transpose, layer_attributes, ref_accept_pruned_input, conv_type):
    default_conv_params = {
        'weight_requires_grad': True,
        'kernel_size': (2, 2),
        'stride': (1, 1),
        'padding_values': [0, 0]
     }
    graph = NNCFGraph()
    dummy_op_before = graph.add_nncf_node('dummy_op_before', DummyMaskProducerMetatype.name,
                                          DummyMaskProducerMetatype)
    target_conv_attributes = ConvolutionLayerAttributes(transpose=transpose, **layer_attributes, **default_conv_params)
    conv_op_target = graph.add_nncf_node('conv_op_target', dummy_types.DummyConvMetatype.name,
                                         dummy_types.DummyConvMetatype,
                                         layer_attributes=target_conv_attributes)
    graph.add_edge_between_nncf_nodes(from_node_id=dummy_op_before.node_id,
                                      to_node_id=conv_op_target.node_id,
                                      tensor_shape=[layer_attributes['in_channels']] * 4,
                                      input_port_id=0,
                                      output_port_id=0,
                                      dtype=Dtype.FLOAT)
    pruning_op_class = dummy_types.DummyOpTransposeConv if transpose else dummy_types.DummyOpConv
    assert pruning_op_class.accept_pruned_input(conv_op_target) == ref_accept_pruned_input
    ones_input_mask = np.ones((layer_attributes['in_channels'],))
    ones_output_mask = np.ones((layer_attributes['out_channels'],))
    for input_mask in [None, ones_input_mask]:
        for output_mask in [None, ones_output_mask]:
            dummy_op_before = graph.get_node_by_id(dummy_op_before.node_id)
            conv_op_target = graph.get_node_by_id(conv_op_target.node_id)
            dummy_op_before.data['output_mask'] = input_mask
            conv_op_target.data['output_mask'] = output_mask
            MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
            dummy_op_before = graph.get_node_by_id(dummy_op_before.node_id)
            conv_op_target = graph.get_node_by_id(conv_op_target.node_id)
            if conv_type == 'usual_conv':
                assert np.all(conv_op_target.data['output_mask'] == output_mask)
            elif conv_type == 'grouped_conv_no_depthwise':
                assert conv_op_target.data['output_mask'] is None
            else:
                assert np.all(conv_op_target.data['output_mask'] == input_mask)



@pytest.mark.parametrize('with_elementwise', [False, True])
def test_stop_ops_elementwise_source_before_concat(with_elementwise):
    graph = NNCFGraph()
    stop_op_0 = graph.add_nncf_node('stop_op_0', 'stop_propagation_ops', dummy_types.DummyStopPropoagtionMetatype)
    stop_op_1 = graph.add_nncf_node('stop_op_1', 'stop_propagation_ops', dummy_types.DummyStopPropoagtionMetatype)
    concat_layer_attributes = MultipleInputLayerAttributes(-1)
    concat_node = graph.add_nncf_node('concat_node', 'concat', dummy_types.DummyConcatMetatype,
                                      layer_attributes=concat_layer_attributes)
    add_node = partial(graph.add_edge_between_nncf_nodes,
                       tensor_shape=[10, 10],
                       input_port_id=0,
                       output_port_id=0,
                       dtype=Dtype.FLOAT)

    if not with_elementwise:
        # stop_op_0 -> concat_node
        add_node(from_node_id=stop_op_0.node_id,
                 to_node_id=concat_node.node_id)

        # stop_op_1 -> concat_node
        add_node(from_node_id=stop_op_1.node_id,
                 to_node_id=concat_node.node_id)
    else:
        elementwise_op = graph.add_nncf_node('elementwise', 'elementwise', dummy_types.DummyElementwiseMetatype)

        # stop_op_0 -> elementwise
        add_node(from_node_id=stop_op_0.node_id,
                 to_node_id=elementwise_op.node_id)

        # stop_op_1 -> elementwise
        add_node(from_node_id=stop_op_1.node_id,
                 to_node_id=elementwise_op.node_id)

        # elementwise -> concat
        add_node(from_node_id=elementwise_op.node_id,
                 to_node_id=concat_node.node_id)

    assert not dummy_types.DummyOpConcat.check_concat(concat_node, graph)
    MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
    concat_node = graph.get_node_by_id(concat_node.node_id)
    assert concat_node.data['output_mask'] is None


@pytest.mark.parametrize('empty_mask_branch', [False, True])
def test_convs_elementwise_source_before_concat(empty_mask_branch):
    graph = NNCFGraph()
    conv_op_0 = graph.add_nncf_node('conv_op_0', 'conv', dummy_types.DummyConvMetatype)
    conv_op_1 = graph.add_nncf_node('conv_op_1', 'conv', dummy_types.DummyConvMetatype)
    conv_op_2 = graph.add_nncf_node('conv_op_2', 'conv', dummy_types.DummyConvMetatype)
    elementwise_node = graph.add_nncf_node('elementwise_node', 'elementwise', dummy_types.DummyElementwiseMetatype)
    concat_layer_attributes = MultipleInputLayerAttributes(2)
    concat_node = graph.add_nncf_node('concat_node', 'concat', dummy_types.DummyConcatMetatype,
                                      layer_attributes=concat_layer_attributes)
    add_node = partial(graph.add_edge_between_nncf_nodes,
                       tensor_shape=[10] * 4,
                       input_port_id=0,
                       output_port_id=0,
                       dtype=Dtype.FLOAT)

    # conv_op_0 -> elementwise_node
    add_node(from_node_id=conv_op_0.node_id,
             to_node_id=elementwise_node.node_id)

    # conv_op_1 -> elementwise_node
    add_node(from_node_id=conv_op_1.node_id,
             to_node_id=elementwise_node.node_id)

    # elementwise_node -> concat_node
    add_node(from_node_id=elementwise_node.node_id,
             to_node_id=concat_node.node_id)

    # conv_op_2 -> concat_node
    add_node(from_node_id=conv_op_2.node_id,
             to_node_id=concat_node.node_id)

    # Check without masks
    assert dummy_types.DummyOpConcat.check_concat(concat_node, graph)
    # Set masks
    masked_convs = [conv_op_0, conv_op_1]
    if not empty_mask_branch:
        masked_convs.append(conv_op_2)

    for conv_op in masked_convs:
        conv_op = graph.get_node_by_id(conv_op.node_id)
        conv_op.data['output_mask'] = np.ones(10)

    # Propagate masks
    MaskPropagationAlgorithm(graph, dummy_types.DUMMY_PRUNING_OPERATOR_METATYPES).mask_propagation()
    # Check with masks
    concat_node = graph.get_node_by_id(concat_node.node_id)
    assert dummy_types.DummyOpConcat.check_concat(concat_node, graph)
    reference_mask = np.ones((20,))
    np.testing.assert_equal(concat_node.data['output_mask'], reference_mask)
