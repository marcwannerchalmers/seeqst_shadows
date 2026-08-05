import pennylane as qp
import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from pennylane.operation import Operator
from typing import List
import math

class Observable(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self) -> Operator:
        pass

    @abstractmethod
    def trace(self) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class PauliObservable(Observable):
    def __init__(self, init_value, sample_indices: List=[0,4],
                 name=None) -> None:
        super().__init__()
        self.mode = "random"
        self.n = 1

        self.pauli_dict = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        self.pauli_array = np.array(list(self.pauli_dict.keys()))
        self.qp_obs_list = [qp.Identity, qp.PauliX, qp.PauliY, qp.PauliZ]
        self.obs_array = None
        self.qubit_obs: List[Operator] =[]
        self.obs = None
        self.obs_string: str = ""
        self.qubit_trace = None
        self.tr = None
        if isinstance(init_value, int):
            self.n = init_value
            if not isinstance(sample_indices[0], int):
                assert len(sample_indices[0] == init_value), "Sample_indices not compatible with observable dimension."
            self.sample(sample_indices)
        elif isinstance(init_value, str):
            self.n = len(init_value)
            self.obs_array = np.array([self.pauli_dict[val] for val in init_value])
            self.obs_string = init_value

        self.name = name if name is not None else "".join(self.pauli_array[self.obs_array])

    def __call__(self)->Operator:
        if self.obs is None:
            self.compute_obs()
        return self.obs

    def compute_obs(self):
        self.obs = qp.prod(*self.qubit_wise_obs())

    def qubit_wise_obs(self):
        if len(self.qubit_obs) == 0:
            self.qubit_obs = [self.qp_obs_list[op](i) for i, op in enumerate(self.obs_array)]
        return self.qubit_obs
    
    def trace(self):
        return math.prod([*self.qubit_wise_trace()])
        
    def qubit_wise_trace(self):
        if self.qubit_trace is None:
            if self.obs is None:
                self.compute_obs()
            self.qubit_trace = [np.trace(op.matrix()) for op in self.qubit_obs]
        return self.qubit_trace

    # sample indices is list of low, high
    def sample(self, sample_indices: List=[0,4]):
        self.obs_array = np.random.randint(*sample_indices,size=(self.n,))
        self.obs_string = sum(self.pauli_dict[self.pauli_array[i]] 
                              for i in range(self.obs_array))
        self.obs = None
        self.qubit_obs = []

    def get_name(self):
        return self.name
