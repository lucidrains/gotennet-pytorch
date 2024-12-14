
from gotennet_pytorch.gotennet import GotenNet

def test_equivariant():

    model = GotenNet(
        dim = 512,
        max_num_degrees = 6,
        depth = 6
    )

    assert True
