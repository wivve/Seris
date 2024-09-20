from typing import List, Optional
from polars.dataframe import DataFrame
from polars.series import Series
from torch import Tensor, tensor


class Tokenizer:
    "character based tokenizer, meaning each char in string have unique token_id"

    def __init__(self, special_token: Optional[List[str]] = None):
        self.kv = dict()
        self.vk = dict()
        if special_token is not None:
            self.__with_special_token(*special_token)

    def __with_special_token(self, *s_token: str) -> None:
        for i, t in enumerate(s_token):
            self.kv[t] = i
        self.__update_vk()

    def __update_vk(self) -> None:
        del self.vk
        self.vk = dict(zip(self.kv.values(), self.kv.keys()))

    @property
    def vocab_size(self) -> int:
        return len(self.kv)

    def train(self, x: DataFrame | Series, index: Optional[str | int] = None) -> None:
        if index is not None and isinstance(x, DataFrame):
            x = x[index].clone()
        assert isinstance(x, Series), "invalid datatype " + repr(type(x))
        _data = set("".join(i for i in x.to_list()))
        for i, j in enumerate(_data, start=self.vocab_size):
            if self.kv.get(j) is None:
                self.kv[j] = i
        self.__update_vk()

    def encode(self, x: str) -> Tensor:
        return tensor([self.kv[i] for i in x])

    def decode(self, x: Tensor) -> List[str]:
        return [self.vk[i] for i in x.detach().tolist()]
