from typing import Any
from abc import ABC, abstractmethod
import gpytorch

import torch
from torch import nn, Tensor

class AcquisitionFun(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    # Objective
    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @abstractmethod
    def add_point(self, x: Tensor):
        pass

# TODO: should contain GP objects and functionality to update it
# Should take some module to compute the objective
class GPAcquisitionFun(AcquisitionFun):
    pass