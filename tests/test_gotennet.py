import pytest
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

from gotennet_pytorch.gotennet import GotenNet, torch_default_dtype

@torch_default_dtype(torch.float64)
def test_invariant():
    model = GotenNet(
        dim = 256,
        max_degree = 2,
        depth = 1,
        heads = 2,
        dim_head = 32,
        dim_edge_refinement = 256,
        return_coors = False,
        ff_kwargs = dict(
            layernorm_input = True
        )
    )

    random_rotation = rot(*torch.randn(3))

    atom_ids = torch.randint(0, 14, (1, 12))
    coors = torch.randn(1, 12, 3)
    adj_mat = torch.randint(0, 2, (1, 12, 12)).bool()
    mask = torch.randint(0, 2, (1, 12)).bool()

    inv1, _ = model(atom_ids, adj_mat = adj_mat, coors = coors, mask = mask)
    inv2, _ = model(atom_ids, adj_mat = adj_mat, coors = coors @ random_rotation, mask = mask)

    assert torch.allclose(inv1, inv2, atol = 1e-5)

@torch_default_dtype(torch.float64)
@pytest.mark.parametrize('num_residual_streams', (1, 4))
def test_equivariant(num_residual_streams):

    model = GotenNet(
        dim = 256,
        max_degree = 2,
        depth = 1,
        heads = 2,
        dim_head = 32,
        dim_edge_refinement = 256,
        return_coors = True,
        ff_kwargs = dict(
            layernorm_input = True
        ),
        num_residual_streams = num_residual_streams
    )

    random_rotation = rot(*torch.randn(3))

    atom_ids = torch.randint(0, 14, (1, 12))
    coors = torch.randn(1, 12, 3)
    adj_mat = torch.randint(0, 2, (1, 12, 12)).bool()
    mask = torch.randint(0, 2, (1, 12)).bool()

    _, coors1 = model(atom_ids, adj_mat = adj_mat, coors = coors, mask = mask)
    _,  coors2 = model(atom_ids, adj_mat = adj_mat, coors = coors @ random_rotation, mask = mask)

    assert torch.allclose(coors1 @ random_rotation, coors2, atol = 1e-5)

@torch_default_dtype(torch.float64)
def test_equivariant_with_atom_feats():

    model = GotenNet(
        dim = 256,
        max_degree = 2,
        depth = 1,
        heads = 2,
        dim_head = 32,
        dim_edge_refinement = 256,
        accept_embed = True,
        return_coors = True
    )

    random_rotation = rot(*torch.randn(3))

    atom_feats = torch.randn((1, 12, 256))
    coors = torch.randn(1, 12, 3)
    adj_mat = torch.randint(0, 2, (1, 12, 12)).bool()
    mask = torch.randint(0, 2, (1, 12)).bool()

    _, coors1 = model(atom_feats, adj_mat = adj_mat, coors = coors, mask = mask)
    _,  coors2 = model(atom_feats, adj_mat = adj_mat, coors = coors @ random_rotation, mask = mask)

    assert torch.allclose(coors1 @ random_rotation, coors2, atol = 1e-5)

@torch_default_dtype(torch.float64)
@pytest.mark.parametrize('num_residual_streams', (1, 4))
def test_equivariant_neighbors(num_residual_streams):

    model = GotenNet(
        dim = 256,
        max_degree = 2,
        depth = 1,
        heads = 2,
        dim_head = 32,
        cutoff_radius = 5.,
        dim_edge_refinement = 256,
        return_coors = True,
        num_residual_streams = num_residual_streams,
        ff_kwargs = dict(
            layernorm_input = True
        )
    )

    random_rotation = rot(*torch.randn(3))

    atom_ids = torch.randint(0, 14, (1, 6))
    coors = torch.randn(1, 6, 3)
    adj_mat = torch.randint(0, 2, (1, 6, 6)).bool()
    mask = torch.randint(0, 2, (1, 6)).bool()

    _, coors1 = model(atom_ids, adj_mat = adj_mat, coors = coors, mask = mask)
    _,  coors2 = model(atom_ids, adj_mat = adj_mat, coors = coors @ random_rotation, mask = mask)

    out1 = coors1 @ random_rotation
    out2 = coors2

    assert torch.allclose(out1, out2, atol = 1e-5)
