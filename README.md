<img src="./gotennet.png" width="450px"></img>

## GotenNet - Pytorch

Implementation of <a href="https://openreview.net/forum?id=5wxCQDtbMo">GotenNet</a>, new SOTA 3d equivariant transformer, in Pytorch

I know a lot of researchers have moved on from geometric learning after Alphafold3. However, I just cannot help but <a href="https://arxiv.org/abs/2410.11443">wonder</a>. Hedging my bets

## Install

```bash
$ pip install gotennet-pytorch
```

## Usage

```python
import torch
from gotennet_pytorch import GotenNet

model = GotenNet(
    dim = 256,
    max_degree = 2,
    depth = 1,
    heads = 2,
    dim_head = 32,
    dim_edge_refinement = 256,
    return_coors = False
)

atom_ids = torch.randint(0, 14, (1, 12))
coors = torch.randn(1, 12, 3)
adj_mat = torch.randint(0, 2, (1, 12, 12)).bool()
lens = torch.randint(1, 12, (1,))

invariant, coors_out = model(atom_ids, adj_mat = adj_mat, coors = coors, lens = lens)
```

## Citations

```bibtex
@inproceedings{anonymous2024rethinking,
    title   = {Rethinking Efficient 3D Equivariant Graph Neural Networks},
    author  = {Anonymous},
    booktitle = {Submitted to The Thirteenth International Conference on Learning Representations},
    year    = {2024},
    url     = {https://openreview.net/forum?id=5wxCQDtbMo},
    note    = {under review}
}
```

```bibtex
@inproceedings{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```
