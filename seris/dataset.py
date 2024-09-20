import polars as pr
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional,List,Callable
from polars.dataframe import DataFrame
from polars.series import Series

class IntoDataset(Dataset):
    def __init__(
        self,
        dataset: DataFrame | Series ,/,
        index : Optional[str] = None,
        mapper: Optional[Callable] = None,
        # data_range : int = None
    ):
        super().__init__()
        if index is not None and isinstance(dataset , DataFrame):
            if mapper is None:
                dataset = dataset.clone()[index]
            else:
                dataset = dataset.clone()[index].map_elements(mapper)
        assert not isinstance(dataset, DataFrame)
        self.dataset = dataset
    
    def __load_from_indecies(*inxs):
        pass
    
    def __getitem__(self, x : int) -> Tensor:
        return self.dataset[x]
        
    def __len__(self):
        return len(self.dataset)

    @classmethod
    def from_parquet(
        cls,
        path: Path | str,/,
        index: Optional[List[str]] = None
    ):
        data: DataFrame = pr.from_parquet(path)
        col = data.columns
        return cls(data , index=col[0])
        
