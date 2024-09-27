import torch
import einops
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass

@dataclass
class ModelConfig:
    dim: int
    lr: float
    d_model: int
    vocab_size: int
    nheads : int
    batch_size : int

class Attention(nn.Module):
    def __init__(
        self,
        d_model: int
    ):
        super().__init__()
        self.d_model = d_model
        self.k_proj = nn.Linear(d_model , d_model)
        self.q_proj = nn.Linear(d_model , d_model)
        self.v_proj = nn.Linear(d_model , d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        k: Tensor = self.k_proj(x)
        q: Tensor = self.q_proj(x)
        v: Tensor = self.v_proj(x)
        return nn.functional.softmax((k@q.transpose(-1,-2)).div(self.d_model),dim=-1)@v

class Head(nn.Module):
    def __init__(
        self,
        dim :int = 64,
        d_model : int = 64,
        nheads : int = 1
    ):
        assert dim%nheads == 0
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim//nheads
        self.heads = nn.ModuleList([Attention(d_model=d_model) for _ in range(nheads)])
        
    def forward(self, x: Tensor) -> Tensor:
        x = einops.rearrange(x , 'b (t c) d -> b t c d' , t=self.nheads)
        x = torch.concat(tuple(head(x[:,i,:]) for i,head in enumerate(self.heads,start=0)),dim=-2)
        return x

class Transformer(nn.Module):
    def __init__(
        self,*,
        config: ModelConfig
    ):
        super().__init__()
        self.dim = config.dim
        self.nheads = config.nheads
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        # self.bs = batch_size
        self.emb_proj = nn.Embedding(self.vocab_size, self.d_model)
        self.multihead = Head( dim = self.dim ,d_model=self.d_model , nheads=self.nheads)
        self.fully_conn = nn.Sequential(
            nn.Linear(self.d_model*self.dim , self.vocab_size)
        )
       
    def forward(self, x: Tensor) -> Tensor:
        x = self.emb_proj( x )
        x = self.multihead( x )
        x = self.fully_conn( x.view(-1, self.d_model*self.dim) )
        return x

    def generate(self , x : Tensor ) -> Tensor:
        y = nn.functional.softmax( self(x) ,dim=-1)
        return torch.multinomial( y , num_samples=1)
