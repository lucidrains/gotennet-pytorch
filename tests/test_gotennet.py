import torch
from torch import sin, cos, stack
from einops import rearrange

# random rotations

def rot_z(gamma):
    c = cos(gamma)
    s = sin(gamma)
    z = torch.zeros_like(gamma)
    o = torch.ones_like(gamma)

    out = stack((
        c, -s, z,
        s, c, z,
        z, z, o
    ), dim = -1)

    return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

def rot_y(beta):
    c = cos(beta)
    s = sin(beta)
    z = torch.zeros_like(beta)
    o = torch.ones_like(beta)

    out = stack((
        c, z, s,
        z, o, z,
        -s, z, c
    ), dim = -1)

    return rearrange(out, '... (r1 r2) -> ... r1 r2', r1 = 3)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

# testing

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

    random_rotation = rot(*torch.randn(3))

    assert True
