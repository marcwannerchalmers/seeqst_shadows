import pennylane as qp
import numpy as np
from abc import ABC, abstractmethod
from pennylane.operation import Operator

class Observable(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self) -> Operator:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class PauliObservable(Observable):
    def __init__(self, init_value) -> None:
        super().__init__()
        self.mode = "random"
        self.n = 1

        self.pauli_dict = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        self.pauli_array = np.array(list(self.pauli_dict.keys()))
        self.qp_obs_list = [qp.Identity, qp.PauliX, qp.PauliY, qp.PauliZ]
        self.obs_array = None
        self.obs = None
        self.obs_string = None
        if isinstance(init_value, int):
            self.n = init_value
            self.sample()
        elif isinstance(init_value, str):
            self.n = len(init_value)
            self.obs_array = np.array([self.pauli_dict[val] for val in init_value])
            self.obs_string = init_value

    def __call__(self)->Operator:
        if self.obs is None:
            self.compute_obs()
        return self.obs

    def compute_obs(self):
        ops = []
        for i, op in enumerate(self.obs_array):
            if op > 0:
                ops.append(self.qp_obs_list[op](i))
        self.obs = qp.prod(*ops)

    def sample(self):
        self.obs_array = np.random.randint(0,4,size=(self.n,))
        self.obs = None

    def get_name(self):
        return "".join(self.pauli_array[self.obs_array])
