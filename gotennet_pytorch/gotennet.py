from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, Tensor, einsum
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
        max_degree,
        mlp_expansion_factor = 2.
    ):
        """
        following eq 13
        """
        super().__init__()
        assert max_degree > 1
        self.max_degree = max_degree

        mlp_dim = int(mlp_expansion_factor * dim * 2)

        self.projs = ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(max_degree)])

        self.mlps = ModuleList([
            Sequential(Linear(dim * 2, mlp_dim), nn.SiLU(), Linear(mlp_dim, dim * 2))
            for _ in range(max_degree)
        ])

    def forward(
        self,
        h: Tensor,
        x: tuple[Tensor, ...]
    ):
        assert len(x) == self.max_degree

        h_residual = 0.
        x_residuals = []

        for one_degree, proj, mlp in zip(x, self.projs, self.mlps):

            # make higher degree tensor invariant through norm on `m` axis and then concat -> mlp -> split

            proj_one_degree = einsum('... d m, ... d e -> ... e m', one_degree, pre_mlp_proj)

            normed_invariant = l2norm(proj_one_degree)

            mlp_inp = torch.cat((x, normed_invariant))
            mlp_out = mlp(mlp_inp)

            m1, m2 = mlp_out.chunk(2, dim = -1) # named m1, m2 in equation 13, one is the residual for h, the other modulates the projected higher degree tensor for its residual

            modulated_one_degree = einx.multiply('... d m, d -> ... d m', proj_one_degree, m2)

            # aggregate residuals

            h_residual = h_residual + m1

            x_residuals.append(modulated_one_degree)

        # handle residuals within the module

        h = h + h_residual
        x = [*map(sum, zip(x, x_residuals))]

        return h, x

# hierarchical tensor refinement
# section 3.4

class HierarchicalTensorRefinement(Module):
    def __init__(
        self,
        dim,
        dim_edge_refinement, # they made this value much higher for MD22 task. so it is an important hparam for certain more difficult tasks
        max_degree,
    ):
        super().__init__()
        assert max_degree > 0

        # in paper, queries share the same projection, but each higher degree has its own key projection

        self.to_queries = nn.Parameter(torch.randn(dim, dim_edge_refinement))

        self.to_keys = ParameterList([nn.Parameter(torch.randn(dim, dim_edge_refinement)) for _ in range])

        # the two weight matrices
        # one for mixing the inner product between all queries and keys across degrees above
        # the other for refining the t_ij passed down from the layer before as a residual

        self.residue_update = nn.Linear(dim, dim, bias = False)
        self.edge_proj = nn.Linear(dim_edge_refinement, dim, bias = False)

    def forward(
        self,
        t_ij: Tensor,
        x: tuple[Tensor, ...],
    ):
        # eq (10)

        queries = [einsum('... d m, ... d e -> ... e m', one_degree, self.to_queries) for one_degree in x]

        keys = [einsum('... d m, ... d e -> ... e m', one_degree, to_keys) for one_degree, to_keys in zip(x, self.to_keys)]

        # eq (11)

        inner_product = [einsum('... i d m, ... j d m -> ... i j d', one_degree_query, one_degree_key) for one_degree_query, one_degree_key in zip(queries, keys)]

        w_ij = cat(inner_product, dim = -1)

        return self.edge_proj(w_ij) + self.residue_update(t_ij) # eq (12)

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
