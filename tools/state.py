import pennylane as qp
from pennylane.typing import TensorLike
import numpy as np
from abc import ABC, abstractmethod
from pennylane.operation import Operator
import qutip as qt
from tools.utils import HChain

# n: number of qubits
class State(ABC):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    @abstractmethod
    def __call__(self) -> Operator:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

class HRState(State):
    def __init__(self, n, seed=12345) -> None:
        super().__init__(n)
        self.state_dm: TensorLike = []
        self.rng = np.random.default_rng(seed)
        self.sample_state()

    # TODO: change seed here too
    def sample_state(self):
        self.state_dm = qt.rand_ket(2**self.n, seed=self.rng).full()[:,0]

    def __call__(self) -> Operator:
        return qp.StatePrep(self.state_dm, wires=range(self.n))



class HChainGS(State):
    def __init__(self, n: int, seed=12345, J="random") -> None:
        super().__init__(n)
        self.rng = np.random.default_rng(seed)
        self.J = J
        self.sample_state()

    def sample_state(self):
        if self.J == "random":
            J = 2*self.rng.uniform(-1,1,(self.n-1,)) - np.ones((self.n-1,))       
        # Can also use a function depending on n and a rng for J
        else:
            J = self.J(self.n, self.rng)

        H = HChain(J)
        _, evec = np.linalg.eigh(H)
        self.state_evec = evec[:,-1]

    def __call__(self) -> Operator:
        return qp.StatePrep(self.state_evec, wires=range(self.n))
        

    