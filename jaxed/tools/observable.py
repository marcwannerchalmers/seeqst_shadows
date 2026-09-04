from flax import struct
from abc import ABC, abstractmethod
from dataclasses import field
import pennylane as qp
from pennylane.operation import Operator
from pennylane import qjit
import jax
from jax import numpy as jnp
from typing import Any, Union, List, ClassVar, Dict
from jax import jit, vmap
from jax import random, Array
from jax.lax import cond
from jax.random import PRNGKey
from functools import partial
from typing import Callable
import time
import qutip as qt
# TODO: Implement flax Observable, which serves as 'model'. Use linen also in the interest of usage with trainable models
# Remove lazy evaluations etc, design as partly a 'translator class' with static methods, partly as model class

class Observable(ABC, struct.PyTreeNode):
    params: Array # needs to be the dimension that it would be if this was an object containing multiple observables
    name_fun: Callable | None = struct.field(pytree_node=False)

    @classmethod
    def init(cls, init_params, name_fun=None):
        return cls(init_params, name_fun)

    @abstractmethod
    def op(self) -> Operator:
        pass

    def circuit(self) -> None:
        pass

    @abstractmethod
    def trace(self) -> Array:
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def _name(cls, param: Array) -> str:
        pass

    def get_name(self) -> List[str]:
        if self.name_fun is None:
            return [self._name(param) for param in self.params]
        else:
            return [self.name_fun(param) for param in self.params]

QP_OBS_LIST = [qp.Identity, qp.PauliX, qp.PauliY, qp.PauliZ]

class PauliObservable(Observable):
    ztype_obs_list: List = struct.field(pytree_node=False)
    pauli_dict: ClassVar[Dict] = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    pauli_list: ClassVar[List] = ["I","X","Y","Z"]
    qp_obs_list: ClassVar[List[Callable[...,Any]]] = QP_OBS_LIST
    matrix_list: ClassVar[Array] = jnp.array([op(0).matrix() for op in QP_OBS_LIST])

    @classmethod
    def init(cls, init_params, name_fun=None):
        params = jnp.stack([cls.get_param(param) for param in init_params])
        n = params.shape[-1]
        ztype_obs_list = list(reversed([qp.Identity(0)] + \
                                    [qp.prod(*[qp.PauliZ(j)  # type: ignore[reportCallIssue]
                                    for j in range(n-1,i-1,-1)]) 
                                    for i in range(n-1,-1,-1)]))

        return cls(params, name_fun=name_fun, ztype_obs_list=ztype_obs_list)
        
    @classmethod
    def init_random(cls, key, sample_indices: List[int]=list(range(4)), n: int=1, N: int=1,
                    name_fun=None, k_local=0, padding_indices: List[int]=[0]):
        if (k_local <= 0) or (k_local > n): 
            k_local = n
        key1, key2, key3 = random.split(key, 3)
        sample_indices = jnp.array(sample_indices)
        sample_params = random.randint(key1, (N,k_local), 0, len(sample_indices))
        params = sample_indices[sample_params]
        if k_local < n:
            padding_indices = jnp.array(padding_indices)
            sample_params2 = random.randint(key2, (N,n-k_local), 0, len(padding_indices))
            params2 = padding_indices[sample_params2]
            params = jnp.concatenate([params, params2], axis=1)
            params = random.permutation(key3, params, axis=1, independent=True)

        ztype_obs_list = list(reversed([qp.Identity(0)] + \
                                    [qp.prod(*[qp.PauliZ(j)  # type: ignore[reportCallIssue]
                                    for j in range(n-1,i-1,-1)]) 
                                    for i in range(n-1,-1,-1)]))

        return cls(params, name_fun=name_fun, ztype_obs_list=ztype_obs_list)

    @classmethod
    def get_param(cls, pauli_str: str):
        obs_array = jnp.array([cls.pauli_dict[val] for val in pauli_str])
        return obs_array 

    def op(self) -> Operator:
        params_onehot = jax.nn.one_hot(self.params, 4)
        n_identity = jnp.dot(params_onehot, jnp.array([1,0,0,0])).sum(axis=-1).astype(int)
        op_param = jax.nn.one_hot(n_identity, self.n+1)
        return qp.sum(*[op_param[i]*self.ztype_obs_list[i] for i in range(self.n+1)]) # type: ignore[reportCallIssue]
    
    def circuit(self) -> None:
        for i in range(self.n):
            if self.params[i] == 1:
                qp.Hadamard(i)
            elif self.params[i] == 2:
                qp.adjoint(qp.S)(i) # type: ignore[reportCallIssue]
                qp.Hadamard(i)

        sorted_indices = jnp.argsort(self.params)
        sorted_params = self.params[sorted_indices]
        for i, idx in enumerate(sorted_indices):
            if i == idx or sorted_params[i] > 0:
                continue
            qp.SWAP(wires=[i, idx])

    def qubit_wise_obs(self) -> List[Operator]:
            params_onehot = jax.nn.one_hot(self.params, 4)

            return [qp.sum(*[params_onehot[i, j] * self.qp_obs_list[j](i) # type: ignore[reportCallIssue]
                                     for j in range(4)])
                                     for i in range(self.n)]

    def trace(self)->Array:
        return jnp.prod(self.qubit_wise_trace())

    def qubit_wise_trace(self)->Array:
        matrices = self.matrix_list[self.params]
        return jnp.einsum("bii->b", matrices)

    # sample indices is list of low, high
    @staticmethod
    def sample(key, n, N, sample_indices: List=[0,4]):
        return jax.random.randint(key, (N,n), *sample_indices)

    @classmethod 
    def obs_string(cls, pauli_array) -> str:
        obs = "".join(cls.pauli_list[pauli_array[i]] for i in range(len(pauli_array)))
        return obs

    @classmethod
    def _name(cls, param: Array) -> str:
        return cls.obs_string(param)

    @property
    def n(self) -> int:
        return self.params.shape[-1]

    @property
    def is_ZType(self) -> Array:
        return jnp.logical_not(jnp.isin(jnp.array([1,2]), self.params).any())

def test_jit():

    n = 5
    N = 10
    pauli_str = "XYZXY"
    name = "X"
    dm = qt.rand_ket(2**n).full()[:,0]

    
    sample_indices = [0,4]
    params = PauliObservable.sample_params(sample_indices, n, N, PRNGKey(1234))
    # params = params.at[0,:,:].set(jnp.array([1,0,0,0]*n).reshape(n,4))
    params = params.at[0,:].set(jnp.array([0]*n))
    print(params)
    outcome = jnp.array([0,1,0,1,0])
    observables = PauliObservable(params=params, n=n)
    print(observables.ztype_obs_list)
    # print(vmap(lambda ob: ob.op())(observables))

    def post_meas_state_gates(outcome):       
        for i, oc in enumerate(outcome):
            if oc == 1:
                qp.X(i)
                
    @qjit(autograph=True)
    @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
    def circuit_rho(obs, outcome):
        qp.StatePrep(dm, wires=range(n))
        obs.circuit()
        return qp.expval(obs.op())
    @qjit(autograph=True)
    @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
    def circ(outcome):
        qp.StatePrep(dm, wires=range(n))
        return qp.expval(qp.PauliY(1) @ qp.PauliY(2))
    
    print("CIrc", circ(outcome))
    t0 = time.perf_counter()
    expvals = vmap(circuit_rho, in_axes=(0, None))(observables, outcome)
    print(expvals)
    t1 = time.perf_counter()

    print(f"first call: {t1 - t0:.6f} s")

    t0 = time.perf_counter()
    expvals = vmap(circuit_rho, in_axes=(0, None))(observables, outcome)
    print(expvals)
    t1 = time.perf_counter()

    print(f"second call: {t1 - t0:.6f} s")

    print(vmap(lambda ob: ob.trace())(observables))
    print(vmap(lambda ob: ob.is_ZType())(observables))

def test_klocal():
    key = PRNGKey(1234)
    obs = PauliObservable.init_random(key, sample_indices=[1,2,3],
                                      n=10, N=20)
    print(obs.params)
    obs2 = PauliObservable.init_random(key, sample_indices=[1,2],
                                          n=10, N=20, k_local=3, padding_indices=[0,3])
    print(obs2.params)

if __name__ == "__main__":
    #test_jit()
    test_klocal()

    """@classmethod
    def qubit_wise_obs(cls):
        if len(self.qubit_obs) == 0:
            self.qubit_obs = [self.qp_obs_list[op](i) for i, op in enumerate(self.obs_array)]
        return self.qubit_obs"""


    """def qubit_wise_trace(self):
            if self.qubit_trace is None:
                if self.obs is None:
                    self.compute_obs()
                self.qubit_trace = [np.trace(op.matrix()) for op in self.qubit_obs]
            return self.qubit_trace"""