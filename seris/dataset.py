from pathlib import Path
from typing import List, Optional, Self

import polars as pr
from torch import Tensor
from torch.utils.data import Dataset


class IntoDataset(Dataset):
    def __init__(
        self,
        dataset ,*,
        index : Optional[str] = None,
    ):
        super().__init__()
        self.dataset = dataset[index]

    def __getitem__(self, x : int) -> Tensor:
        return self.dataset[x]
        
    def __len__(self):
        return len(self.dataset)

    @classmethod
    def from_parquet(
        cls,
        path: Path | str,*,
        index: Optional[List[str]] = None
    ) -> Self:
        data = pr.read_parquet(path).select([
            pr.concat_str( index ).alias('data')
        ])
        return cls(data , index='data')
        

class ParquteLoader():

    def __init__(self, path:str , col :list[str] = ['input', 'output']) -> None:
        self.path = path
        self.data = pr.read_parquet(path).select([
            pr.concat_str( col ).alias("data")
        ]).to_dict()

    def __iter__(self):
        return iter(self.data['data'])

class SimpleDataset(Dataset):
    def __init__(self , path):
        super().__init__()
        self.path = path
        self.data = pr.read_parquet(path).select(
            pr.concat_str(['input', 'output']).alias('merge')
        ).to_dict()['merge']
    def __getitem__(self , x):
        return self.data[x]
    def __len__(self):
        return len(self.data)




class GetData(Dataset):

    def __init__(self , ca : str):
        super().__init__()
        self.data = ca.split("\n")

    def __getitem__(self , x):
        return self.data[x]

    def __len__(self):
        return len(self.data)
