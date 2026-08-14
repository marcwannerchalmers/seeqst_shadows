from abc import ABC, abstractmethod
import torch
from torch import Tensor

class Domain(ABC):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    # negative if x is in domain
    @abstractmethod
    def distance(self, x: Tensor)->Tensor:
        pass

    @abstractmethod
    def project_on_boundary(self, x: Tensor)->Tensor:
        pass

class Rectangle(Domain):
    pass

class Simplex(Domain):
    pass