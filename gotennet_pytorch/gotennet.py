from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, Tensor, einsum
from torch.nn import Linear, Sequential, Module, ModuleList, ParameterList

import einx
from einops import reduce
from einops.layers.torch import Rearrange

from e3nn.o3 import spherical_harmonics

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

            proj_one_degree = einsum('... d m, ... d e -> ... e m', one_degree, proj)

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

        self.to_keys = ParameterList([nn.Parameter(torch.randn(dim, dim_edge_refinement)) for _ in range(max_degree)])

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

# geometry-aware tensor attention
# section 3.3

class GeometryAwareTensorAttention(Module):
    def __init__(
        self,
        dim,
        max_degree,
        dim_head = None,
        heads = 8,
        mlp_expansion_factor = 2.
    ):
        super().__init__()
        assert max_degree > 0
        self.max_degree = max_degree

        dim_head = default(dim_head, dim)

        # for some reason, there is no mention of attention heads, will just improvise

        dim_inner = dim * dim_head

        self.split_heads = Rearrange('b ... (h d) -> b h ... d', h = heads)
        self.merge_heads = Rearrange('b h ... d -> b ... (h d)')

        # eq (5) - layernorms are present in the diagram in figure 2. but not mentioned in the equations..

        self.to_hi = nn.LayerNorm(dim, bias = False)
        self.to_hj = nn.LayerNorm(dim, bias = False)

        self.to_queries = Linear(dim, dim_inner, bias = False)
        self.to_keys = Linear(dim, dim_inner, bias = False)

        dim_mlp_inner = int(mlp_expansion_factor * dim_inner)

        # S contains two parts of L_max (one to modulate each degree of r_ij, another to modulate each X_j, then one final to modulate h). incidentally, this overlaps with eq. (m = 2 * L + 1), causing much confusion, cleared up in openreview

        self.S = (1, max_degree, max_degree)
        S = sum(self.S)

        self.to_values = Sequential(
            Linear(dim, dim_mlp_inner),
            nn.SiLU(),
            Linear(dim_mlp_inner, S * dim_inner),
            Rearrange('... (s d) -> ... s d', s = S)
        )

        # eq (6) second half: t_ij -> edge scalar features

        self.to_edge_keys = Sequential(
            Linear(dim, S * dim_inner, bias = False),  # W_re
            nn.SiLU(),                                 # σ_k - never indicated in paper. just use Silu
            Rearrange('... (s d) -> ... s d', s = S)
        )

        # eq (7) - todo, handle cutoff radius

        self.to_edge_values = nn.Sequential(           # t_ij modulating weighted sum
            Linear(dim, S * dim_inner, bias = False),
            Rearrange('... (s d) -> ... s d', s = S)
        )

        self.post_attn_h_values = Sequential(
            Linear(dim, dim_mlp_inner),
            nn.SiLU(),
            Linear(dim_mlp_inner, S * dim_inner),
            Rearrange('... (s d) -> ... s d', s = S)
        )

        # combine heads

        self.combine_heads = Sequential(
            Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        h: Tensor,
        x: tuple[Tensor, ...],
        t_ij: Tensor,
        r_ij: tuple[Tensor, ...]
    ):
        assert len(x) == self.max_degree
        assert len(r_ij) == self.max_degree

        # eq (5)

        hi = self.to_hi(h)
        hj = self.to_hj(h)

        queries = self.to_queries(hi)

        keys = self.to_keys(hj)
        values = self.to_values(hj)

        post_attn_values = self.post_attn_h_values(hj)

        edge_keys = self.to_edge_values(t_ij)

        edge_values = self.to_edge_values(t_ij)

        # split out attention heads

        queries, keys, values, post_attn_values, edge_keys, edge_values = map(self.split_heads, (queries, keys, values, post_attn_values, edge_keys, edge_values))

        # eq (6)

        # numerator - αij

        sim_num = einsum('... i d, ... j d -> ... i j', queries, keys)

        # denominator - αik (?) - why is there a k-dimension? just play along for now

        sim_den = einsum('... i, ... k s -> ... i s', queries, edge_keys)

        # attention

        attn = einx.divide('... i j, ... i s -> ... i j s', sim_num, sim_den)

        # aggregate values

        sea_ij = einsum('... i j s, ... j s d -> ... i j s d', attn, values)

        # eq (7)

        sea_ij = sea_ij + einx.multiply('... i j s d, ... j s d -> ... i j s d', edge_values, post_attn_values)

        # combine heads - not in paper for some reason, but attention heads mentioned, so must be necessary?

        out = self.merge_heads(sea_ij)

        out = self.combine_heads(out)

        # split out all the O's (eq 7 second half)

        h_scales, r_ij_scales, x_scales = out.chunk(self.S, dim = -2)

        # modulate with invariant scales and sum residuals

        h_with_residual = h + reduce(h_scales, '... j d -> ... d', 'sum')

        x_with_residual = []

        for one_degree, one_r_ij, one_degree_scale, one_r_ij_scale in zip(x, r_ij, x_scales, r_ij_scales):

            x_with_residual.append((
                einx.multiply('... j d m, ... i j d -> ... i d m', one_degree, one_degree_scale) +
                einx.multiply('... i j m, ... i j d -> ... i d m', one_r_ij, one_r_ij_scale)
            ))

        return h_with_residual, x_with_residual

# main class

class GotenNet(Module):
    def __init__(
        self,
        dim,
        depth,
        max_degree,
        dim_edge_refinement,
        heads = 8,
        dim_head = None,
        mlp_expansion_factor = 2.,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        assert max_degree > 0
        self.max_degree = max_degree

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                HierarchicalTensorRefinement(dim, dim_edge_refinement, max_degree),
                GeometryAwareTensorAttention(dim, max_degree, dim_head, heads, mlp_expansion_factor),
                EquivariantFeedForward(dim, max_degree, mlp_expansion_factor),
            ]))

    def forward(
        self,
        atom_ids,
        coors,
        mask = None
    ):

        rel_pos = einx.subtract('b i c, b j c -> b i j c')
        rel_dist = rel_dir.norm(dim = -1)

        # constitute r_ij from section 3.1

        r_ij = []

        for degree in range(1, self.max_degree + 1):
            one_degree_r_ij = spherical_harmonics(degree, rel_pos, normalize = True)
            r_ij.append(one_degree_r_ij)

        # go through the layers

        for htr, attn, ff in self.layers:
            pass
  
        return x, coors
