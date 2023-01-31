from nncf.quantization.quantize import quantize
from tests.torch.helpers import TwoConvTestModel


def test_nncf_graph_building():
    model = TwoConvTestModel()
    quantize(model, )
    NNCFGraphFactory.create()
    pass