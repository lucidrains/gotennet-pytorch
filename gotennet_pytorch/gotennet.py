from __future__ import annotations

from functools import partial
from contextlib import contextmanager
from collections.abc import Sequence

import torch
from torch import nn, cat, Tensor, einsum
from torch.nn import Linear, Sequential, Module, ModuleList, ParameterList

import einx
from einops import reduce
from einops.layers.torch import Rearrange

from e3nn.o3 import spherical_harmonics

from gotennet_pytorch.tensor_typing import Float, Int, Bool

# ein notation

# b - batch
# h - heads
# n - sequence
# i, j - source and target sequence
# d - feature
# m - order of each degree
# l - degree

# constants

LayerNorm = partial(nn.LayerNorm, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def mask_from_lens(lens, total_len):
    seq = torch.arange(total_len, device = lens.device)
    return einx.less('n, b -> b n', seq, lens)

@contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# radial basis function

class Radial(Module):
    def __init__(
        self,
        dim,
        radial_hidden_dim = 64
    ):
        super().__init__()

        hidden = radial_hidden_dim

        self.rp = Sequential(
            Rearrange('... -> ... 1'),
            Linear(1, hidden),
            nn.SiLU(),
            LayerNorm(hidden),
            Linear(hidden, hidden),
            nn.SiLU(),
            LayerNorm(hidden),
            Linear(hidden, dim)
        )

    def forward(self, x):
        return self.rp(x)

# node scalar feat init
# eq (1) and (2)

class NodeScalarFeatInit(Module):
    def __init__(
        self,
        num_atoms,
        dim,
        accept_embed = False,
        radial_hidden_dim = 64
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(num_atoms, dim) if not accept_embed else nn.Linear(dim, dim)
        self.neighbor_atom_embed = nn.Embedding(num_atoms, dim) if not accept_embed else nn.Linear(dim, dim)

        self.rel_dist_mlp = Radial(
            dim = dim,
            radial_hidden_dim = radial_hidden_dim
        )

        self.to_node_feats = Sequential(
            Linear(dim * 2, dim),
            LayerNorm(dim),
            nn.SiLU(),
            Linear(dim, dim)
        )

    def forward(
        self,
        atoms: Int['b n'] | Float['b n d'],
        rel_dist: Float['b n n'],
        adj_mat: Bool['b n n'] | None = None,
        mask: Bool['b n'] | None = None,
        radius_cutoff_softmask: Float['b n n'] | None = None
    ) -> Float['b n d']:

        dtype = rel_dist.dtype
        batch, seq, device = *atoms.shape[:2], atoms.device

        eye = torch.eye(seq, device = device, dtype = torch.bool)

        if exists(adj_mat):
            adj_mat = adj_mat & ~eye # remove self from adjacency matrix
        else:
            adj_mat = torch.ones((batch, seq, seq), device = device, dtype = dtype)

        embeds = self.atom_embed(atoms)

        rel_dist_feats = self.rel_dist_mlp(rel_dist)

        if exists(mask):
            rel_dist_feats = einx.where('b j, b i j d, -> b i j d', mask, rel_dist_feats, 0.)

        neighbor_embeds = self.neighbor_atom_embed(atoms)

        if exists(radius_cutoff_softmask):
            rel_dist_feats = einx.multiply('b i j d, b i j -> b i j d', rel_dist_feats, radius_cutoff_softmask)

        neighbor_feats = einsum('b i j, b i j d, b j d -> b i d', adj_mat.type(dtype), rel_dist_feats, neighbor_embeds)

        self_and_neighbor = torch.cat((embeds, neighbor_feats), dim = -1)

        return self.to_node_feats(self_and_neighbor)

# edge scalar feat init
# eq (3)

class EdgeScalarFeatInit(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.,
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor)

        self.rel_dist_mlp = Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, dim_inner, bias = False),
            LayerNorm(dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        h: Float['b n d'],
        rel_dist: Float['b n n']
    ) -> Float['b n n d']:

        outer_sum_feats = einx.add('b i d, b j d -> b i j d', h, h)

        rel_dist_feats = self.rel_dist_mlp(rel_dist)

        return outer_sum_feats + rel_dist_feats

# cutoff function
# never expounded upon in the paper. just improvise a smooth version

class SoftCutoff(Module):
    def __init__(
        self,
        cutoff_dist = 5., # they have it at about 4-5 angstroms
        sharpness = 10.   # multiply value before sigmoid
    ):
        super().__init__()
        self.cutoff_dist = cutoff_dist
        self.sharpness = sharpness

    def forward(
        self,
        rel_dist: Float['b n n']
    ) -> Float['b n n']:

        shifted_and_scaled = (rel_dist - self.cutoff_dist) * self.sharpness

        modulation = shifted_and_scaled.sigmoid()

        return 1. - modulation

# equivariant feedforward
# section 3.5

class EquivariantFeedForward(Module):
    def __init__(
        self,
        dim,
        max_degree,
        mlp_expansion_factor = 2.,
        layernorm_input = False # no mention of this in the paper, but think there should be one based on my own intuition
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
            Sequential(
                LayerNorm(dim * 2) if layernorm_input else nn.Identity(),
                Linear(dim * 2, mlp_dim),
                nn.SiLU(),
                Linear(mlp_dim, dim * 2)
            )
            for _ in range(max_degree)
        ])

    def forward(
        self,
        h: Float['b n d'],
        x: Sequence[Float['b n d _'], ...]
    ):
        assert len(x) == self.max_degree

        h_residual = 0.
        x_residuals = []

        for one_degree, proj, mlp in zip(x, self.projs, self.mlps):

            # make higher degree tensor invariant through norm on `m` axis and then concat -> mlp -> split

            proj_one_degree = einsum('... d m, ... d e -> ... e m', one_degree, proj)

            normed_invariant = proj_one_degree.norm(dim = -1)

            mlp_inp = torch.cat((h, normed_invariant), dim = - 1)
            mlp_out = mlp(mlp_inp)

            m1, m2 = mlp_out.chunk(2, dim = -1) # named m1, m2 in equation 13, one is the residual for h, the other modulates the projected higher degree tensor for its residual

            modulated_one_degree = einx.multiply('... d m, ... d -> ... d m', proj_one_degree, m2)

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
        self.edge_proj = nn.Linear(dim_edge_refinement * max_degree, dim, bias = False)

    def forward(
        self,
        t_ij: Float['b n n d'],
        x: Sequence[Float['b n d _'], ...],
    ) -> Float['b n n d']:

        # eq (10)

        queries = [einsum('... d m, ... d e -> ... e m', one_degree, self.to_queries) for one_degree in x]

        keys = [einsum('... d m, ... d e -> ... e m', one_degree, to_keys) for one_degree, to_keys in zip(x, self.to_keys)]

        # eq (11)

        inner_product = [einsum('... i d m, ... j d m -> ... i j d', one_degree_query, one_degree_key) for one_degree_query, one_degree_key in zip(queries, keys)]

        w_ij = cat(inner_product, dim = -1)

        # eq (12)

        edge_proj_out = self.edge_proj(w_ij)
        residue_update_out = self.residue_update(t_ij)

        return edge_proj_out + residue_update_out

# geometry-aware tensor attention
# section 3.3

class GeometryAwareTensorAttention(Module):
    def __init__(
        self,
        dim,
        max_degree,
        dim_head = None,
        heads = 8,
        mlp_expansion_factor = 2.,
        only_init_high_degree_feats = False # if set to True, only returns high degree steerable features eq (4) in section 3.2
    ):
        super().__init__()
        self.only_init_high_degree_feats = only_init_high_degree_feats

        assert max_degree > 0
        self.max_degree = max_degree

        dim_head = default(dim_head, dim)

        # for some reason, there is no mention of attention heads, will just improvise

        dim_inner = heads * dim_head

        self.split_heads = Rearrange('b ... (h d) -> b h ... d', h = heads)
        self.merge_heads = Rearrange('b h ... d -> b ... (h d)')

        # eq (5) - layernorms are present in the diagram in figure 2. but not mentioned in the equations..

        self.to_hi = LayerNorm(dim)
        self.to_hj = LayerNorm(dim)

        self.to_queries = Linear(dim, dim_inner, bias = False)
        self.to_keys = Linear(dim, dim_inner, bias = False)

        dim_mlp_inner = int(mlp_expansion_factor * dim_inner)

        # S contains two parts of L_max (one to modulate each degree of r_ij, another to modulate each X_j, then one final to modulate h). incidentally, this overlaps with eq. (m = 2 * L + 1), causing much confusion, cleared up in openreview

        self.S = (1, max_degree, max_degree) if not only_init_high_degree_feats else (max_degree,)
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

        # alphafold styled gating

        self.to_gates = Sequential(
            Linear(dim, heads * S),
            nn.Sigmoid(),
            Rearrange('b i (h s) -> b h i 1 s 1', s = S),
        )

        # combine heads

        self.combine_heads = Sequential(
            Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        h: Float['b n d'],
        t_ij: Float['b n n d'],
        r_ij: Sequence[Float['b n n _'], ...],
        x: Sequence[Float['b n d _'], ...] | None = None,
        mask: Bool['b n'] | None = None,
        radius_cutoff_softmask: Float['b n n'] | None = None
    ):
        # validation

        assert exists(x) ^ self.only_init_high_degree_feats

        if not self.only_init_high_degree_feats:
            assert len(x) == self.max_degree

        assert len(r_ij) == self.max_degree

        # eq (5)

        hi = self.to_hi(h)
        hj = self.to_hj(h)

        queries = self.to_queries(hi)

        keys = self.to_keys(hj)

        # unsure why values are split into two, with one elementwise-multiplied with the edge values coming from t_ij
        # need another pair of eyes to double check

        values = self.to_values(hj)

        post_attn_values = self.post_attn_h_values(hj)

        edge_keys = self.to_edge_keys(t_ij)
        edge_values = self.to_edge_values(t_ij)

        # split out attention heads

        queries, keys, values, post_attn_values, edge_keys, edge_values = map(self.split_heads, (queries, keys, values, post_attn_values, edge_keys, edge_values))

        # eq (6)

        # unsure why there is a k-dimension in the paper math notation, in addition to i, j

        keys = einx.multiply('... j d, ... i j s d -> ... i j s d', keys, edge_keys)

        # sim

        sim = einsum('... i d, ... i j s d -> ... i j s', queries, keys)

        # masking

        if exists(mask):
            sim = einx.where('b j, b h i j s, -> b h i j s', mask, sim, max_neg_value(sim))

        # attn

        attn = sim.softmax(dim = -2)

        # aggregate values

        sea_ij = einsum('... i j s, ... j s d -> ... i j s d', attn, values)

        # apply soft radius cutoff to projected features off t_ij from the hierarchical tensor refinemnt module

        if exists(radius_cutoff_softmask):
            edge_values = einx.multiply('b h i j s d, b i j -> b h i j s d', edge_values, radius_cutoff_softmask)

        # eq (7)

        sea_ij = sea_ij + einx.multiply('... i j s d, ... j s d -> ... i j s d', edge_values, post_attn_values)

        # alphafold style gating

        out = sea_ij * self.to_gates(hi)

        # combine heads - not in paper for some reason, but attention heads mentioned, so must be necessary?

        out = self.merge_heads(out)

        out = self.combine_heads(out)

        # maybe eq (4) and early return

        if self.only_init_high_degree_feats:
            return [einsum('... i j m, ... i j d -> ... i d m', one_r_ij, one_r_ij_scale) for one_r_ij, one_r_ij_scale in zip(r_ij, out.unbind(dim = -2))]

        # split out all the O's (eq 7 second half)

        h_scales, r_ij_scales, x_scales = out.split(self.S, dim = -2)

        # modulate with invariant scales and sum residuals

        h_with_residual = h + reduce(h_scales, 'b i j 1 d -> b i d', 'sum')
        x_with_residual = []

        for one_degree, one_r_ij, one_degree_scale, one_r_ij_scale in zip(x, r_ij, x_scales.unbind(dim = -2), r_ij_scales.unbind(dim = -2)):

            x_with_residual.append((
                einsum('b j d m, b i j d -> b i d m', one_degree, one_degree_scale) +
                einsum('b i j m, b i j d -> b i d m', one_r_ij, one_r_ij_scale)
            ))

        return h_with_residual, x_with_residual

# main class

class GotenNet(Module):
    def __init__(
        self,
        dim,
        depth,
        max_degree,
        dim_edge_refinement = None,
        accept_embed = False,
        num_atoms = 14,
        heads = 8,
        dim_head = None,
        mlp_expansion_factor = 2.,
        edge_init_mlp_expansion_factor = 4.,
        cutoff_radius = 5.,
        ff_kwargs: dict = dict(),
        cutoff_kwargs: dict = dict(),
        return_coors = True,
        proj_invariant_dim = None,
        final_norm = True
    ):
        super().__init__()
        self.accept_embed = accept_embed

        assert max_degree > 0
        self.max_degree = max_degree

        dim_edge_refinement = default(dim_edge_refinement, dim)

        # soft cutoff

        self.soft_cutoff = SoftCutoff(cutoff_radius, **cutoff_kwargs)

        # node and edge feature init

        self.node_init = NodeScalarFeatInit(num_atoms, dim, accept_embed = accept_embed)
        self.edge_init = EdgeScalarFeatInit(dim, expansion_factor = edge_init_mlp_expansion_factor)

        self.high_degree_init = GeometryAwareTensorAttention(
            dim,
            max_degree = max_degree,
            dim_head = dim_head,
            heads = heads,
            mlp_expansion_factor = mlp_expansion_factor,
            only_init_high_degree_feats = True
        )

        # layers, thus deep learning

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                HierarchicalTensorRefinement(dim, dim_edge_refinement, max_degree),
                GeometryAwareTensorAttention(dim, max_degree, dim_head, heads, mlp_expansion_factor),
                EquivariantFeedForward(dim, max_degree, mlp_expansion_factor),
            ]))

        # not mentioned in paper, but transformers need a final norm

        self.final_norm = final_norm

        if final_norm:
            self.h_final_norm = nn.LayerNorm(dim)

            self.x_final_norms = ModuleList([Sequential(Rearrange('... d m -> ... m d'), LayerNorm(dim), Rearrange('... m d -> ... d m')) for _ in range(max_degree)])

        # maybe project invariant

        self.proj_invariant = None

        if exists(proj_invariant_dim):
            self.proj_invariant = Linear(dim, proj_invariant_dim, bias = False)

        # maybe project to coordinates

        self.proj_to_coors = Sequential(
            Rearrange('... d m -> ... m d'),
            Linear(dim, 1, bias = False),
            Rearrange('... 1 -> ...')
        ) if return_coors else None

    def forward(
        self,
        atoms: Int['b n'] | Float['b n d'],
        coors: Float['b n 3'],
        adj_mat: Bool['b n n'] | None = None,
        lens: Int['b'] | None = None,
        mask: Bool['b n'] | None = None
    ):
        assert (atoms.dtype in (torch.int, torch.long)) ^ self.accept_embed

        batch, seq_len = atoms.shape[:2]

        assert not (exists(lens) and exists(mask)), '`lens` and `masks` cannot be both passed in'

        if exists(lens):
            mask = mask_from_lens(lens, seq_len)

        rel_pos = einx.subtract('b i c, b j c -> b i j c', coors, coors)
        rel_dist = rel_pos.norm(dim = -1)

        # soft cutoff

        radius_cutoff_softmask = self.soft_cutoff(rel_dist)

        # initialization

        h = self.node_init(atoms, rel_dist, adj_mat, mask = mask, radius_cutoff_softmask = radius_cutoff_softmask)

        t_ij = self.edge_init(h, rel_dist)

        # constitute r_ij from section 3.1

        r_ij = []

        for degree in range(1, self.max_degree + 1):
            one_degree_r_ij = spherical_harmonics(degree, rel_pos, normalize = True, normalization = 'norm')
            r_ij.append(one_degree_r_ij)

        # init the high degrees

        x = self.high_degree_init(h, t_ij, r_ij, mask = mask)

        # go through the layers

        for htr, attn, ff in self.layers:

            # hierarchical tensor refinement

            t_ij = htr(t_ij, x)

            # followed by attention, but of course

            h, x = attn(h, t_ij, r_ij, x, mask = mask, radius_cutoff_softmask = radius_cutoff_softmask)

            # feedforward

            h, x = ff(h, x)
  
        # maybe final norms

        if self.final_norm:
            h = self.h_final_norm(h)
            x = [norm(one_degree) for one_degree, norm in zip(x, self.x_final_norms)]

        # maybe transform invariant h

        if exists(self.proj_invariant):
            h = self.proj_invariant(h)

        # return h and x if `return_coors = False`

        if not exists(self.proj_to_coors):
            return h, x

        degree1, *_ = x

        coors_out = self.proj_to_coors(degree1)

        return h, coors_out
