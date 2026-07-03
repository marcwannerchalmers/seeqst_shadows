import pennylane as qp
from pennylane.typing import TensorLike
import numpy as np
from abc import ABC, abstractmethod
from pennylane.operation import Operator
import qutip as qt

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
    def __init__(self, n) -> None:
        super().__init__(n)
        self.state_dm: TensorLike = []
        self.sample_state()

    def sample_state(self):
        self.state_dm = qt.rand_ket(2**self.n).full()[:,0]

    def __call__(self) -> Operator:
        return qp.StatePrep(self.state_dm, wires=range(self.n))