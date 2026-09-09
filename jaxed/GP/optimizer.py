import jax
from jax import Array, vmap
from flax.struct import dataclass, field
from flax import linen as nn
from abc import ABC, abstractmethod

# TODO: Think about baseoptimizer for hyperparams
# Should optimize params, as well as hyperparams
# Should lead to training loop as in e.g. pytorch
class ParamOptimizer(ABC, nn.Module):
    @abstractmethod
    def step(self):
        pass


# TODO: Implement ParamOptimizer that iteratively solves
# the matrix problem and trains the hyperparams
class ExactOptimizer(ParamOptimizer):
    # give it some baseoptimizer
    pass