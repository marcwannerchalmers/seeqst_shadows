from __future__ import annotations
import pennylane as qp
from pennylane.typing import TensorLike
import numpy as np
from abc import ABC, abstractmethod
from pennylane.operation import Operator
import qutip as qt
from jaxed.tools.utils import HChain
import jax 
from jax import random, Array
from jax.random import PRNGKey
from jax import numpy as jnp
from flax import struct

# Could also have a separate sampler for the parameters,
# however this seems to be an overkill for now.

# n: number of qubits
class State(ABC, struct.PyTreeNode):
    n: int = struct.field(pytree_node=False)

    @classmethod
    @abstractmethod
    def sample(cls, n, key, *args, **argv)->State:
        pass

    @classmethod
    @abstractmethod
    def create(cls, n, *args, **argv)->State:
        pass

    @abstractmethod
    def __call__(self) -> Operator:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

# New convention: Need to call sample to sample multiple random instances
# Hence, states can be accessed deterministically or randomly via sample
# This class of states can only be accessed randomly
class HRState(State):

    key: Array
    state_dm: Array

    @classmethod
    def sample(cls, n: int, key: Array):
        state = cls.sample_state(n, key)
        return cls(n=n, key=key, state_dm=state)

    @classmethod
    def create(cls, n, *args, **argv) -> State:
        raise NotImplementedError("Can only be sampled")

    @staticmethod
    def sample_state(n, key: Array):
        key_r, key_i = jax.random.split(key)

        z = (
            jax.random.normal(key_r, (2**n,))
            + 1j * jax.random.normal(key_i, (2**n,))
        )

        return z / jnp.linalg.norm(z)

    def __call__(self) -> Operator:
        return qp.StatePrep(self.state_dm, wires=range(self.n))


class HChainGS(State):
    state_dm: Array

    @classmethod
    def sample(cls, n: int, key: Array, lower: float=-1, upper: float=1):
        J = random.uniform(key, (n-1,), lower, upper)
        return cls.create(n, J)

    @classmethod
    def create(cls, n, J):
        H = HChain(J).matrix()
        _, evec = jnp.linalg.eigh(H)
        state = evec[:,-1]
        return cls(n=n, state_dm=state)

    def __call__(self) -> Operator:
        return qp.StatePrep(self.state_dm, wires=range(self.n))
        

    