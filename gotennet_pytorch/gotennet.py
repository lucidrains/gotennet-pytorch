from __future__ import annotations

import torch
from torch import nn, Tensor, einsum
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Module, ModuleList, ParameterList

import einx
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# equivariant feedforward
# section 3.5

class EquivariantFeedForward(Module):
    def __init__(
        self,
        dim,
        max_num_degrees,
        mlp_expansion_factor = 2.
    ):
        """
        following eq 13
        """
        super().__init__()
        assert max_num_degrees > 0

        mlp_dim = int(mlp_expansion_factor * dim * 2)

        self.proj = ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(max_num_degrees)])

        self.pre_mlp_proj = ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(max_num_degrees)])

        self.mlps = ModuleList([
            Sequential(Linear(dim * 2, mlp_dim), nn.SiLU(), Linear(mlp_dim, dim * 2))
            for _ in range(max_num_degrees - 1)
        ])

    def forward(
        self,
        h: Tensor,
        x: tuple[Tensor, ...]
    ):

        h_residual = 0.
        x_residuals = []

        for one_degree, proj, pre_mlp_proj, mlp in zip(x, self.projs, self.pre_mlp_projs, self.mlps):

            # make higher degree tensor invariant through norm on `m` axis and then concat -> mlp -> split

            proj_one_degree_for_mlp = einsum('... d m, ... d e -> ... e m', one_degree, pre_mlp_proj)

            normed_invariant = l2norm(proj_one_degree_for_mlp)

            mlp_inp = torch.cat((x, normed_invariant))
            mlp_out = mlp(mlp_inp)

            m1, m2 = mlp_out.chunk(2, dim = -1) # named m1, m2 in equation 13, one is the residual for h, the other modulates the projected higher degree tensor for its residual

            # in a stray comment on openreview, W_vu can be two separate weight matrices for expressivity

            proj_one_degree = einsum('... d m, ... d e -> ... e m', one_degree, proj)

            modulated_one_degree = einx.multiply('... d m, d -> ... d m', proj_one_degree, m2)

            # aggregate residuals

            h_residual = h_residual + m1

            x_residuals.append(modulated_one_degree)

        # handle residuals within the module

        h = h + h_residual
        x = [*map(sum, zip(x, x_residuals))]

        return h, x

# main class

class GotenNet(Module):
    def __init__(
        self,
        dim,
        depth,
        max_num_degrees,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        assert max_num_degrees > 0

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                EquivariantFeedForward(dim, max_num_degrees, **ff_kwargs)
            ]))

    def forward(
        self,
        x,
        coors
    ):
        return x, coors
