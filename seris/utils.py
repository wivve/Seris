import torch
from torch import Tensor

def get_batch_rand(tensor_data:Tensor, batch_size=32, dim=64):
    # ( batchsize*32 , ( input*64 ) )  , ( batchsize*32 , ( output*1 ))
    length = len(tensor_data) -1
    rnd = torch.randint( length - dim ,(batch_size,))
    x = torch.concat( [ tensor_data[i:i+dim].unsqueeze(0) for i in rnd ] ,dim=0)
    y = torch.concat( [ tensor_data[i+dim].unsqueeze(0) for i in rnd ] ,dim=0)
    return x,y

def get_batch(tensor_data:Tensor, batch_size=32, dim=64):
    # ( batchsize*32 , ( input*64 ) )  , ( batchsize*32 , ( output*1 ))
    length = len(tensor_data) -batch_size
    for i in range(length-dim):
        lst = list(range(i,i+batch_size))
        x_ = torch.concat( [ tensor_data[i:i+dim].unsqueeze(0) for i in lst ] ,dim=0)
        y_ = torch.concat( [ tensor_data[i+dim].unsqueeze(0) for i in lst ] ,dim=0)
        yield x_,y_
