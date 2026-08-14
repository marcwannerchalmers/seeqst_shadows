from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from GP.domain import Domain
from GP.acquisition import AcquisitionFun


class OnlineAlgo(ABC):
    def __init__(self, acquisition_fun: AcquisitionFun,
                 domain: Domain) -> None:
        super().__init__()
        self.acquisition_fun = acquisition_fun
        self.domain = domain

    def run(self, N: int):
        for i in range(N):
            new_point = self.optimize()
            self.acquisition_fun.add_point(new_point)

    @abstractmethod
    def optimize(self)->Tensor:
        pass
    