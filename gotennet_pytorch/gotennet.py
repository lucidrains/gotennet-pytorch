from __future__ import annotations

from functools import partial
from contextlib import contextmanager
from collections.abc import Sequence

import torch
from torch import nn, cat, Tensor, einsum
from torch.nn import Linear, Sequential, Module, ModuleList, ParameterList

import einx
from einx import get_at

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from e3nn.o3 import spherical_harmonics

from gotennet_pytorch.tensor_typing import Float, Int, Bool

from hyper_connections import get_init_and_expand_reduce_stream_functions

# ein notation

# b - batch
# h - heads
# n - sequence
# m - sequence (neighbors)
# i, j - source and target sequence
# d - feature
# m - order of each degree
# l - degree

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

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

@contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)

# normalization

LayerNorm = partial(nn.LayerNorm, bias = False)

class HighDegreeNorm(Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        norms = x.norm(dim = -1, keepdim = True)
        den = norms.norm(dim = -2, keepdim = True) * self.scale
        return x / den.clamp(min = self.eps) * self.gamma

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
        rel_dist: Float['b n m'],
        adj_mat: Bool['b n m'] | None = None,
        neighbor_indices: Int['b n m'] | None = None,
        neighbor_mask: Bool['b n m'] | None = None,
        mask: Bool['b n'] | None = None,
    ) -> Float['b n d']:

        dtype = rel_dist.dtype
        batch, seq, device = *atoms.shape[:2], atoms.device

        if not exists(adj_mat):
            adj_mat = torch.ones_like(rel_dist, device = device, dtype = dtype)

        embeds = self.atom_embed(atoms)

        rel_dist_feats = self.rel_dist_mlp(rel_dist)

        if exists(neighbor_indices):
            if exists(neighbor_mask):
                rel_dist_feats = einx.where('b i j, b i j d, -> b i j d', neighbor_mask, rel_dist_feats, 0.)
        else:
            if exists(mask):
                rel_dist_feats = einx.where('b j, b i j d, -> b i j d', mask, rel_dist_feats, 0.)

        neighbor_embeds = self.neighbor_atom_embed(atoms)

        if exists(neighbor_indices):
            neighbor_embeds = get_at('b [n] d, b i j -> b i j d', neighbor_embeds, neighbor_indices)

            neighbor_feats = einsum('b i j, b i j d, b i j d -> b i d', adj_mat.type(dtype), rel_dist_feats, neighbor_embeds)
        else:
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
        rel_dist: Float['b n m'],
        neighbor_indices: Int['b n m'] | None = None
    ) -> Float['b n m d']:

        if exists(neighbor_indices):
            h_neighbors = get_at('b [n] d, b i j -> b i j d', h, neighbor_indices)

            outer_sum_feats = einx.add('b i d, b i j d -> b i j d', h, h_neighbors)
        else:
            outer_sum_feats = einx.add('b i d, b j d -> b i j d', h, h)

        rel_dist_feats = self.rel_dist_mlp(rel_dist)

        return outer_sum_feats + rel_dist_feats

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

        # return residuals

        return h_residual, x_residuals

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
        t_ij: Float['b n m d'],
        x: Sequence[Float['b n d _'], ...],
        neighbor_indices: Int['b n m'] | None = None,
    ) -> Float['b n m d']:

        # eq (10)

        queries = [einsum('... d m, ... d e -> ... e m', one_degree, self.to_queries) for one_degree in x]

        keys = [einsum('... d m, ... d e -> ... e m', one_degree, to_keys) for one_degree, to_keys in zip(x, self.to_keys)]

        # eq (11)

        if exists(neighbor_indices):
            keys = [get_at('b [n] d m, b i j -> b i j d m', one_degree_key, neighbor_indices) for one_degree_key in keys]

            inner_product = [einsum('... i d m, ... i j d m -> ... i j d', one_degree_query, one_degree_key) for one_degree_query, one_degree_key in zip(queries, keys)]
        else:
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
        softclamp_value = 50.,
        mlp_expansion_factor = 2.,
        only_init_high_degree_feats = False, # if set to True, only returns high degree steerable features eq (4) in section 3.2
        learned_value_residual_mix = False,
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

        # attention softclamping, used in Gemma

        self.softclamp = partial(softclamp, value = softclamp_value)

        # S contains two parts of L_max (one to modulate each degree of r_ij, another to modulate each X_j, then one final to modulate h). incidentally, this overlaps with eq. (m = 2 * L + 1), causing much confusion, cleared up in openreview

        self.S = (1, max_degree, max_degree) if not only_init_high_degree_feats else (max_degree,)
        S = sum(self.S)

        self.to_values = Sequential(
            Linear(dim, dim_mlp_inner),
            nn.SiLU(),
            Linear(dim_mlp_inner, S * dim_inner),
            Rearrange('... (s d) -> ... s d', s = S)
        )

        # value residual, iclr 2024 paper that is certain to take off

        self.to_value_residual_mix = Sequential(
            Linear(dim, heads, bias = False),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1 1')
        ) if learned_value_residual_mix else None

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
        t_ij: Float['b n m d'],
        r_ij: Sequence[Float['b n m _'], ...],
        x: Sequence[Float['b n d _'], ...] | None = None,
        mask: Bool['b n'] | None = None,
        neighbor_indices: Int['b n m'] | None = None,
        neighbor_mask: Bool['b n m'] | None = None,
        return_value_residual = False,
        value_residuals: Tuple[Tensor, Tensor] | None = None
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

        # edge keys and values

        edge_keys = self.to_edge_keys(t_ij)
        edge_values = self.to_edge_values(t_ij)

        # split out attention heads

        queries, keys, values, post_attn_values, edge_keys, edge_values = map(self.split_heads, (queries, keys, values, post_attn_values, edge_keys, edge_values))

        # value residual mixing

        next_value_residuals = (values, post_attn_values, edge_values)

        if exists(self.to_value_residual_mix):
            assert exists(value_residuals)

            value_residual, post_attn_values_residual, edge_values_residual = value_residuals

            mix = self.to_value_residual_mix(hi)

            values = values.lerp(value_residual, mix)
            post_attn_values = post_attn_values.lerp(post_attn_values_residual, mix)

            if exists(neighbor_indices):
                mix = get_at('b h [n] ..., b i j -> b h i j ...', mix, neighbor_indices)
            else:
                mix = rearrange(mix, 'b h j ... -> b h 1 j ...')

            edge_values = edge_values.lerp(edge_values_residual, mix)

        # account for neighbor logic

        if exists(neighbor_indices):
            keys = get_at('b h [n] ..., b i j -> b h i j ...', keys, neighbor_indices)
            values = get_at('b h [n] ..., b i j -> b h i j ...', values, neighbor_indices)
            post_attn_values = get_at('b h [n] ..., b i j -> b h i j ...', post_attn_values, neighbor_indices)

        # eq (6)

        # unsure why there is a k-dimension in the paper math notation, in addition to i, j

        if exists(neighbor_indices):
            keys = einx.multiply('... i j d, ... i j s d -> ... i j s d', keys, edge_keys)
        else:
            keys = einx.multiply('... j d, ... i j s d -> ... i j s d', keys, edge_keys)

        # similarities

        sim = einsum('... i d, ... i j s d -> ... i j s', queries, keys)

        # soft clamping - used successfully in gemma to prevent attention logit overflows

        sim = self.softclamp(sim)

        # masking

        if exists(neighbor_indices):
            if exists(neighbor_mask):
                sim = einx.where('b i j, b h i j s, -> b h i j s', neighbor_mask, sim, max_neg_value(sim))
        else:
            if exists(mask):
                sim = einx.where('b j, b h i j s, -> b h i j s', mask, sim, max_neg_value(sim))

        # attend

        attn = sim.softmax(dim = -2)

        # aggregate values

        if exists(neighbor_indices):
            sea_ij = einsum('... i j s, ... i j s d -> ... i j s d', attn, values)
        else:
            sea_ij = einsum('... i j s, ... j s d -> ... i j s d', attn, values)

        # eq (7)

        if exists(neighbor_indices):
            sea_ij = sea_ij + einx.multiply('... i j s d, ... i j s d -> ... i j s d', edge_values, post_attn_values)
        else:
            sea_ij = sea_ij + einx.multiply('... i j s d, ... j s d -> ... i j s d', edge_values, post_attn_values)

        # alphafold style gating

        out = sea_ij * self.to_gates(hi)

        # combine heads - not in paper for some reason, but attention heads mentioned, so must be necessary?

        out = self.merge_heads(out)

        out = self.combine_heads(out)

        # maybe eq (4) and early return

        if self.only_init_high_degree_feats:
            x_ij_init = [einsum('... i j m, ... i j d -> ... i d m', one_r_ij, one_r_ij_scale) for one_r_ij, one_r_ij_scale in zip(r_ij, out.unbind(dim = -2))]
            return x_ij_init

        # split out all the O's (eq 7 second half)

        h_scales, r_ij_scales, x_scales = out.split(self.S, dim = -2)

        # modulate with invariant scales and sum residuals

        h_residual = reduce(h_scales, 'b i j 1 d -> b i d', 'sum')
        x_residuals = []

        for one_degree, one_r_ij, one_degree_scale, one_r_ij_scale in zip(x, r_ij, x_scales.unbind(dim = -2), r_ij_scales.unbind(dim = -2)):

            r_ij_residual = einsum('b i j m, b i j d -> b i d m', one_r_ij, one_r_ij_scale)

            if exists(neighbor_indices):
                one_degree_neighbors = get_at('b [n] d m, b i j -> b i j d m', one_degree, neighbor_indices)

                x_ij_residual = einsum('b i j d m, b i j d -> b i d m', one_degree_neighbors, one_degree_scale)

            else:
                x_ij_residual = einsum('b j d m, b i j d -> b i d m', one_degree, one_degree_scale)

            x_residuals.append(r_ij_residual + x_ij_residual)

        out = (h_residual, x_residuals)

        if not return_value_residual:
            return out

        return out, next_value_residuals

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
        cutoff_radius = None,
        max_neighbors = float('inf'),
        mlp_expansion_factor = 2.,
        edge_init_mlp_expansion_factor = 4.,
        ff_kwargs: dict = dict(),
        return_coors = True,
        proj_invariant_dim = None,
        final_norm = True,
        add_value_residual = True,
        num_residual_streams = 4
    ):
        super().__init__()
        self.accept_embed = accept_embed

        assert max_degree > 0
        self.max_degree = max_degree

        dim_edge_refinement = default(dim_edge_refinement, dim)

        # hyper connections, applied to invariant h for starters

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # only consider neighbors less than `cutoff_radius`, in paper, they used ~ 5 angstroms
        # can further randomly select from eligible neighbors with `max_neighbors`

        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors

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
        self.residual_fns = ModuleList([])

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(ModuleList([
                HierarchicalTensorRefinement(dim, dim_edge_refinement, max_degree),
                GeometryAwareTensorAttention(dim, max_degree, dim_head, heads, mlp_expansion_factor, learned_value_residual_mix = add_value_residual and not is_first),
                EquivariantFeedForward(dim, max_degree, mlp_expansion_factor),
            ]))

            self.residual_fns.append(ModuleList([
                init_hyper_conn(dim = dim),
                init_hyper_conn(dim = dim),
            ]))

        # not mentioned in paper, but transformers need a final norm

        self.final_norm = final_norm

        if final_norm:
            self.h_final_norm = nn.LayerNorm(dim)

            self.x_final_norms = ModuleList([HighDegreeNorm(dim) for _ in range(max_degree)])

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

        batch, seq_len, device = *atoms.shape[:2], atoms.device

        assert not (exists(lens) and exists(mask)), '`lens` and `masks` cannot be both passed in'

        if exists(lens):
            mask = mask_from_lens(lens, seq_len)

        rel_pos = einx.subtract('b i c, b j c -> b i j c', coors, coors)
        rel_dist = rel_pos.norm(dim = -1)

        # process adjacency matrix

        if exists(adj_mat):
            eye = torch.eye(seq_len, device = device, dtype = torch.bool)
            adj_mat = adj_mat & ~eye # remove self from adjacency matrix

        # figure out neighbors, if needed

        neighbor_indices: Int['b n m'] | None = None
        neighbor_mask: Bool['b n m'] | None = None

        if exists(self.cutoff_radius):

            if exists(mask):
                rel_dist = einx.where('b j, b i j, -> b i j', mask, rel_dist, 1e6)

            is_neighbor = (rel_dist <= self.cutoff_radius).float()

            max_eligible_neighbors = is_neighbor.sum(dim = -1).long().amax().item()
            max_neighbors = min(max_eligible_neighbors, self.max_neighbors)

            noised_is_neighbor = is_neighbor + torch.rand_like(is_neighbor) * 1e-3
            neighbor_indices = noised_is_neighbor.topk(k = max_neighbors, dim = -1).indices

            if exists(adj_mat):
                adj_mat = adj_mat.gather(-1, neighbor_indices)

            neighbor_dist = rel_dist.gather(-1, neighbor_indices)
            neighbor_mask = neighbor_dist <= self.cutoff_radius

            rel_dist = neighbor_dist
            rel_pos = rel_pos.gather(-2, repeat(neighbor_indices, '... -> ... c', c = 3))

        # initialization

        h = self.node_init(atoms, rel_dist, adj_mat, mask = mask, neighbor_indices = neighbor_indices, neighbor_mask = neighbor_mask)

        t_ij = self.edge_init(h, rel_dist, neighbor_indices = neighbor_indices)

        # constitute r_ij from section 3.1

        r_ij = []

        for degree in range(1, self.max_degree + 1):
            one_degree_r_ij = spherical_harmonics(degree, rel_pos, normalize = True, normalization = 'norm')
            r_ij.append(one_degree_r_ij)

        # init the high degrees

        x = self.high_degree_init(h, t_ij, r_ij, mask = mask, neighbor_indices = neighbor_indices, neighbor_mask = neighbor_mask)

        # value residual

        value_residuals = None

        # maybe expand invariant h residual stream

        h = self.expand_streams(h)

        # go through the layers

        for (htr, attn, ff), (h_attn_residual_fn, h_ff_residual_fn) in zip(self.layers, self.residual_fns):

            # hierarchical tensor refinement

            t_ij = htr(t_ij, x, neighbor_indices = neighbor_indices)

            # followed by attention, but of course

            h, add_attn_residual = h_attn_residual_fn(h)

            (h_residual, x_residuals), next_value_residuals = attn(h, t_ij, r_ij, x, mask = mask, neighbor_indices = neighbor_indices, neighbor_mask = neighbor_mask, value_residuals = value_residuals, return_value_residual = True)

            # add attention residuals

            h = add_attn_residual(h_residual)
            x = [*map(sum, zip(x, x_residuals))]

            # handle value residual

            value_residuals = default(value_residuals, next_value_residuals)

            # feedforward

            h, add_ff_residual = h_ff_residual_fn(h)

            h_residual, x_residuals = ff(h, x)

            # add feedforward residuals

            h = add_ff_residual(h_residual)
            x = [*map(sum, zip(x, x_residuals))]
  
        h = self.reduce_streams(h)

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
