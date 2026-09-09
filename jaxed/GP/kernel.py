from typing import Any
from abc import ABC, abstractmethod

import jax
from jax import Array, vmap
from flax.struct import dataclass, field
from flax import linen as nn

class Kernel(nn.Module, ABC):
    @abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        pass

    def matrix(self, x: Array, y: Array): 
        return vmap(vmap(self.__call__, 
                         in_axes=(None, 0)),
                         in_axes=(0, None)
                         )(x,y)