from abc import ABC, abstractmethod
from typing import List, Any
from tools.utils import build_parallel_entangler_blocks, post_meas_state_gates, parse_circuit, HadjS
from tools.observable import Observable, PauliObservable
from tools.state import State, HRState
import pennylane as qp
from pennylane.typing import TensorLike
import numpy as np
from numpy.typing import NDArray, ArrayLike
from pennylane.operation import Operator
from qiskit.quantum_info.random import random_clifford
from tools.estimator import Estimator, MedianOfMeans

large_width = 400
np.set_printoptions(linewidth=large_width)

################### Prototype of shadow class ###################

class Shadow(ABC):
    # n: number of qubits
    # gate_indices: indices the U(.) method can take. Can also give as a (multidim-) range [min, max] (latter is default)
    def __init__(self, state: State, gate_indices: List, estimator: Estimator=Estimator()) -> None:
        super().__init__()
        self.n = state.n
        self.gate_indices = gate_indices
        self.indices = np.array([])
        self.outcomes = []
        self.rho = state
        self.estimator = estimator

    # create shadow from N state samples
    def create(self, N: int):
        self.sample_indices(N)
        N = len(self.indices)
        # self.outcomes = np.zeros(N,)
        for i in range(N):
            self.outcomes.append(self.sample_circuit(self.indices[i]).T)

    def predict(self, obs_list)->NDArray:    
        preds = self.compute_preds(obs_list)
        return self.estimator(preds)

    def compute_preds(self, obs_list):
        N = len(self.indices)
        preds = np.zeros((len(obs_list), N))
        for idx, outcome in enumerate(self.outcomes):
            oc = self.prop_inverse_measurement(outcome, idx, obs_list)
            preds[:, idx] = oc

        return preds

    def state(self):
        self.rho()

    @abstractmethod
    def U(self, ind: List[int])->None:
        pass          

    # default is simply using a random int
    def sample_indices(self, N: int):
        self.indices = np.random.randint(*self.gate_indices, size=(N,))
        return self.indices

    # returns the properties of the inverse measurements as np array
    @abstractmethod
    def prop_inverse_measurement(self, outcome, ind, obs_list)->NDArray:
        pass

    def sample_circuit(self, ind: List[int]):
        @qp.set_shots(1)
        @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
        def circuit(ind: List[int]):
            self.state()
            self.U(ind)
            return qp.sample(wires=list(range(self.n)))

        return circuit(ind)
    
    def ground_truth(self, obs_list):
        gts = np.zeros((len(obs_list),))
        @qp.qnode(qp.device("lightning.qubit"))
        def circuit(obs):
            self.state()
            return qp.expval(obs)
        
        for i, obs in enumerate(obs_list):
            gts[i] = circuit(obs())

        return gts
    
    def rho_pm(self, outcome, ind):
        post_meas_state_gates(outcome, one_ev=1)
        qp.adjoint(self.U)(self.indices[ind])

class SEEQSTShadow(Shadow):
    # block_idx is the index of the block, whose 
    # binary representation correpsonds to the subset U is sampled from
    # Should be a bitstring or so
    def __init__(self, state: State, gate_indices: List, estimator: Estimator=Estimator(),
                 full_setting=False) -> None:
        super().__init__(state, gate_indices, estimator)
        self.full_setting = full_setting
    
    def sample_indices(self, N: int):
        init_indices = super().sample_indices(N)
        if self.full_setting:
            a1 = np.stack([init_indices, np.zeros_like(init_indices)], axis=1)
            a2 = np.stack([init_indices, np.ones_like(init_indices)], axis=1)
            self.indices = np.concatenate([a1,a2])
        else:
            a1 = np.random.randint(0,2, size=init_indices.shape)
            self.indices = np.stack([init_indices, a1], axis=1)

        return self.indices

    def U(self, ind: ArrayLike)->None:
        block_idx, xy = ind
        sel_circ_text = build_parallel_entangler_blocks(block_idx, self.n, xy)      
        parse_circuit("".join(sel_circ_text))

    # Using b_i as binary representation of 0 <= i < 2**n
    # formula: 2(\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>) - Id/2**n \
    # + 2**(n+1) ( \rho - (\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>)).

    def prop_inverse_measurement(self, outcome, ind, obs_list: List[Observable],
                                 test_mode=False) -> NDArray:
        estimates = np.zeros((len(obs_list,)))

        for i, obs in enumerate(obs_list):
            if not isinstance(obs, PauliObservable):
                raise NotImplementedError()
                
            
            @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
            def circuit_rho(outcome):
                post_meas_state_gates(outcome, one_ev=1)
                qp.adjoint(self.U)(self.indices[ind])
                return qp.expval(obs())
            
            @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
            def circuit_sum(outcome):
                post_meas_state_gates(outcome, one_ev=1)
                qp.adjoint(self.U)(self.indices[ind])
                return qp.probs()

            # TODO: Implement non-Pauli case
            """probs = circuit_sum(outcome)

            sum_value = 0
            for x, p in enumerate(probs):
                sum_value += p * np.real(qp.matrix(obs())[x,x])
                print(sum_value)"""
            # Migrate adapted to Z-type observable here
            exp = 0 if obs.is_ZType() else self.n
            estimates[i] = 2**(exp+1)*circuit_rho(outcome) # + (2-2**(self.n+1))*sum_value  # assuming Pauli observable
            if test_mode and np.abs(estimates[i]) > 1e-10:
                print(self.indices[ind], estimates[i])

        return estimates
    
class PauliShadow(Shadow):
    def __init__(self, state: State, gate_indices: List=[], estimator: Estimator=Estimator()) -> None:
        super().__init__(state, gate_indices, estimator)
        self.Uis = [qp.Hadamard, HadjS(), qp.Identity]

    def sample_indices(self, N: int):
        self.indices = np.random.randint(0,3, size=(N, self.n))
        return self.indices
        
    def U(self, ind: ArrayLike):
        for i in range(self.n):
            self.Uis[ind[i]](i)

    def prop_inverse_measurement(self, outcome, ind, obs_list: List[PauliObservable],
                                 test_mode=False) -> NDArray:
        
        estimates = np.zeros((len(obs_list,)))
        
        for i, obs in enumerate(obs_list):
            @qp.qnode(qp.device("default.qubit", wires=range(self.n)))
            def circuit_rho(outcome):
                self.rho_pm(outcome, ind)
                return [qp.expval(oi) for oi in obs.qubit_wise_obs()]
            estimates[i] = np.prod(3*np.array(circuit_rho(outcome)) - np.real(np.array(obs.qubit_wise_trace())))

            if test_mode and np.abs(estimates[i]) > 1e-10:
                print(self.indices[ind], estimates[i])

        return estimates         

class CliffordShadow(Shadow):
    def __init__(self, state: State, gate_indices: List=[], estimator: Estimator=Estimator()) -> None:
        super().__init__(state, gate_indices, estimator)
        self.Us: List[Operator] = []

    def sample_indices(self, N: int):
        self.Us = []
        for _ in range(N):
            circ = random_clifford(self.n).to_circuit()
            self.Us.append(qp.from_qiskit(circ))
        self.indices = np.arange(N)[:, None]
        return self.indices

    def U(self, ind: ArrayLike):
        self.Us[ind[0]]()

    def prop_inverse_measurement(self, outcome, ind, obs_list: List[Observable],
                                 test_mode=False) -> NDArray:
        
        estimates = np.zeros((len(obs_list,)))
        for i, obs in enumerate(obs_list):
            @qp.qnode(qp.device("default.qubit", wires=range(self.n)))
            def circuit_rho(outcome):
                self.rho_pm(outcome, ind)
                return qp.expval(obs())
    
            estimates[i] = (2**self.n + 1)*circuit_rho(outcome) - np.real(obs.trace())

            if test_mode and np.abs(estimates[i]) > 1e-10:
                print(f" out:", self.indices[ind], estimates[i])

        return estimates

def test_seeqst_antidiagonal():
    obs = PauliObservable("XYYY")
    state = HRState(4)
    shadow = SEEQSTShadow(state, [0,2**4])
    # ind = [15,1]
    for i in range(16):
        for j in range(2):
            ind = [i,j]
            shadow.indices = np.array([ind])
            for outcome in np.ndindex((2,2,2,2)):
                shadow.prop_inverse_measurement(outcome, 0, [obs], test_mode=True)
    # outcome = np.array([0,1,1,0])

def test_pauli_shadow():
    obs = PauliObservable("XXXX")
    state = HRState(4)
    shadow = PauliShadow(state)
    for ind in np.ndindex((3,3,3,3)):
        shadow.indices = np.array([ind])
        for outcome in np.ndindex((2,2,2,2)):
            shadow.prop_inverse_measurement(outcome, 0, [obs], test_mode=True)


def test_clifford_shadow():
    obs = PauliObservable("XXXX")
    state = HRState(4)
    shadow = CliffordShadow(state)
    N = 100
    for ind in range(N):
        shadow.sample_indices(1)
        for outcome in np.ndindex((2,2,2,2)):
            print(f"ind: {ind}, outcome: {outcome}")
            shadow.prop_inverse_measurement(outcome, 0, [obs], test_mode=True)

def test_correlations():
    obs = PauliObservable("XXII")
    N = 100
    state = HRState(4)
    shadow = PauliShadow(state)
    shadow.create(N)
    res1 = shadow.predict([obs])
    # get basis/outcome of first and second qubit and apply formula #2*b[idx]+oc[idx]-1
    # apply the shadow_alignment function to this. Then compute mean. Compare

    mns = [2*shadow.indices[:,i]*np.ones((N,)) + np.array(shadow.outcomes)[:,i,0]
                            for i in range(2)]

    def shadow_alignment(m, n):
        if m > n:
            return shadow_alignment(n, m)  # make sure m is smaller than n

        if m == n and n <= 1:
            return 9  # same basis and outcome
        elif m % 2 == 0 and m == n - 1 and n <= 1:
            # measurments (0,1), (2,3), (4,5),
            return -9  # same basis but different outcome
        else:
            return 0
    shadows = []
    ms, ns = mns

    for i in range(N):
        shadows.append(shadow_alignment(ms[i], ns[i]))
    print(np.array(shadow.outcomes).shape)
    for i in range(N):
        print(shadows[i], shadow.estimates[i], shadow.outcomes[i][0:2].T, shadow.indices[i], ms[i], ns[i])
    res2 = sum(shadows)/N
    print("Pred1: ", res1, "Pred prev: ", res2)

if __name__ == "__main__":
    #test_seeqst_antidiagonal()
    # test_pauli_shadow()
    # test_clifford_shadow()
    test_correlations()
    

    

