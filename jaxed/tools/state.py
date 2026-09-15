from __future__ import annotations
import pennylane as qp
from pennylane.typing import TensorLike
import numpy as np
from abc import ABC, abstractmethod
from pennylane.operation import Operator
import qutip as qt
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from jaxed.tools.utils import HChain, build_parallel_entangler_blocks
from jaxed.tools.clifford import GHZ_type_state_clifford, Tableau
import jax 
from jax import random, Array, vmap, jit
from jax.random import PRNGKey
from jax import numpy as jnp
from flax import struct
from functools import partial

# Could also have a separate sampler for the parameters,
# however this seems to be an overkill for now.

# n: number of qubits
class State(ABC, struct.PyTreeNode):

    @classmethod
    @abstractmethod
    def init_random(cls, key: Array, N_state: int, n: int, *args, **argv)->State:
        pass

    @classmethod
    @abstractmethod
    def init(cls, *args, **argv)->State:
        pass

    @abstractmethod
    def __call__(self) -> None:
        pass

    @property
    @abstractmethod
    def n(self)->int:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

# New convention: Need to call sample to sample multiple random instances
# Hence, states can be accessed deterministically or randomly via sample
# This class of states can only be accessed randomly
class HRState(State):

    state_dm: Array

    @classmethod
    def init_random(cls, key: Array, N_state: int, n: int):
        keys = random.split(key, N_state)
        state = vmap(cls.sample_state, in_axes=(0, None))(keys, n)
        return cls(state_dm=state)

    @classmethod
    def init(cls, statevecs: Array) -> State:
        return cls(state_dm=jnp.asarray(statevecs, dtype=jnp.complex64))

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def sample_state(key: Array, n):
        key_r, key_i = jax.random.split(key)

        z = (
            jax.random.normal(key_r, (2**n,), dtype=jnp.float32)
            + 1j * jax.random.normal(key_i, (2**n,), dtype=jnp.float32)
        )

        return (z / jnp.linalg.norm(z)).astype(jnp.complex64)

    @property
    def n(self) -> int:
        return int(np.log2(self.state_dm.shape[-1]))

    def __call__(self):
        qp.StatePrep(self.state_dm, wires=range(self.n))


class HChainGS(State):
    state_dm: Array

    @classmethod
    def init_random(cls, key: Array, n: int, lower: float=-1, upper: float=1):
        J = random.uniform(key, (n-1,), dtype=jnp.float32, minval=lower, maxval=upper)
        return cls.init(J)

    @classmethod
    def init(cls, J: Array):
        H = jnp.asarray(HChain(J).matrix(), dtype=jnp.complex64)
        _, evec = jnp.linalg.eigh(H)
        state = evec[:,-1]
        return cls(state_dm=state)

    @property
    def n(self) -> int:
        return int(np.log2(self.state_dm.shape[-1]))

    def __call__(self):
        qp.StatePrep(self.state_dm, wires=range(self.n))

class GHZType(State):
    selective_block: Array
    xy: Array

    @classmethod
    def init_random(cls, key: Array, N_state: int, n: int) -> State:
        blocks = random.randint(key, (N_state, n+1), 0, 2)
        return vmap(cls.init)(blocks[:,:n], blocks[:,n])

    @classmethod
    def init(cls, selective_block: Array, xy: Array) -> State:
        return cls(selective_block=selective_block, xy=xy)

    @property
    def n(self) -> int:
        return int(self.selective_block.shape[-1])

    def __call__(self) -> None:
        build_parallel_entangler_blocks(self.selective_block,
                                        self.n, 
                                        self.xy)

    def clifford(self, tableau: Tableau):
        return GHZ_type_state_clifford(self.selective_block, 
                                       self.xy,
                                       tableau)

def test():
    n = 5
    state = HRState.init_random(jax.random.PRNGKey(1234), n)
    @qp.qjit
    @qp.set_shots(1)
    @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
    def circuit(state: State):
        state()
        # U(ind)
        return qp.init_random(wires=list(range(state.n)))

    

    print(circuit(state))

if __name__ == "__main__":
    test()
