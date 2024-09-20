import torch
import einops
import torch.nn as nn
from torch import Tensor

class Attention(nn.Module):
    def __init__(
        self,
        head_dim: int
    ):
        super().__init__()
        self.head_dim = head_dim
        self.k_proj = nn.Linear(head_dim , head_dim)
        self.q_proj = nn.Linear(head_dim , head_dim)
        self.v_proj = nn.Linear(head_dim , head_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        k: Tensor = self.k_proj(x)
        q: Tensor = self.q_proj(x)
        v: Tensor = self.v_proj(x)
        return nn.functional.softmax((k@q.transpose(-1,-2)).div(self.head_dim),dim=-1)@v

class Head(nn.Module):
    def __init__(
        self,
        d_model : int = 64,
        nheads : int = 1
    ):
        assert d_model%nheads == 0
        super().__init__()
        self.nheads = nheads
        self.head_dim = d_model//nheads
        self.heads = nn.ModuleList([Attention(head_dim=self.head_dim) for _ in range(nheads)])
        
    def forward(self, x: Tensor) -> Tensor:
        x = einops.rearrange(x , 'b c (t d) -> b c t d' , t=self.nheads)
        x = torch.concat(tuple(head(x[:,:,i]) for i,head in enumerate(self.heads)),dim=-1)
        return x

class Transformer(nn.Module):
    def __init__(
        self,/,
        nheads: int = 1,
        d_model: int =32,
        vocab_size: int = 1
    ):
        super().__init__()
        self.nheads = nheads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.emb_proj = nn.Embedding(vocab_size, d_model)
        self.multihead = Head( d_model=d_model , nheads=nheads)
        self.fully_conn = nn.Sequential(
            nn.Linear(d_model , vocab_size)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.emb_proj( x )
        x = self.multihead( x )
        x = self.fully_conn( x )
        return x
