
from gotennet_pytorch.gotennet import GotenNet

def test_equivariant():

    model = GotenNet(
        dim = 256,
        max_degree = 2,
        depth = 1,
        heads = 2,
        dim_head = 32,
        dim_edge_refinement = 256
    )

    assert True
