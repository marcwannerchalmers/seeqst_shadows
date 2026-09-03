import torch
from torch import Tensor
import os
from torch.utils.data import Dataset, DataLoader
from jax import Array
# from tensordict import TensorDict
import copy
import numpy as np

# dataset object
# shadow_size=0 <--> exact correlations
class Data(Dataset):
    def __init__(self, 
                 indices: Array,
                 outcomes: Array) -> None:
        
        super().__init__()
        self.indices = Tensor(indices)
        self.outcomes = Tensor(outcomes)

    def __len__(self):
        return len(self.outcomes)

    # Can add a function here that maps the data given index
    def __getitem__(self, index: int):
        outcome = self.outcomes[index]
        outcome = (-1)**(outcome.sum(dim=-1))

        return self.indices[index]+1, outcome

# returns dataloader objects for training set
def get_train_set(indices: Array,
                  outcomes: Array,
                  batch_size=16) -> DataLoader:

    train_set = Data(indices, outcomes)

    return  DataLoader(train_set, 
                      batch_size=batch_size, 
                      shuffle=True)

def main():
    pass

if __name__ == "__main__":
    main()