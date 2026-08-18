from abc import ABC, abstractmethod
from typing import List, Any, ClassVar, Callable

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxed.tools.utils import build_parallel_entangler_blocks, \
                              post_meas_state_gates, HadjS, PartialCircuit, \
                              create_tableau, canonical_form
from jaxed.tools.observable import Observable, PauliObservable
from jaxed.tools.state import State, HRState
import pennylane as qp
from pennylane import qjit
from pennylane.typing import TensorLike
import numpy as np
from numpy.typing import NDArray, ArrayLike
from pennylane.operation import Operator
from jaxed.tools.estimator import Estimator, MedianOfMeans
import jax 
from jax import numpy as jnp
from jax import Array, vmap, random
from flax import struct
from jax.lax import cond
from functools import partial

large_width = 400
np.set_printoptions(linewidth=large_width)


# To vectorize, can have an enumeration of relevant objects (including identity if equal length is needed)
# and broadcast over vector with indices
# Add key to this class
class Shadow(ABC, struct.PyTreeNode):
    # n: number of qubits
    # sample_idx_range: indices the U(.) method can take. Can also give as a (multidim-) range [min, max] (latter is default)
    rho: State
    estimator: Estimator
    sample_idx_range: Array = struct.field(pytree_node=False)
    n: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    indices: Array
    outcomes: Array
    U: PartialCircuit
    Udag: PartialCircuit

    # create shadow from N state samples
    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               sample_idx_range: Array=jnp.array([]), 
               estimator: Estimator=Estimator(),
               *args, **kwargs):
        
        indices = cls.sample_indices(key, (N,), sample_idx_range,
                                     *args, **kwargs)
        kwargs.pop('n', None)
        kwargs.pop('N', None)
        static_args, dynamic_args = cls.get_U_params(N=N, n=rho.n, **kwargs)

        if not hasattr(cls, "Udag_fun"):
            cls.Udag_fun = qp.adjoint(cls.U_fun)

        instance = cls(rho=rho, estimator=estimator, 
                   sample_idx_range=sample_idx_range, n=rho.n, 
                   N=N, indices=indices, outcomes=jnp.zeros((N,rho.n)),
                   U=PartialCircuit(cls.U_fun, static_args={**static_args}, dynamic_args={**dynamic_args}),
                   Udag=PartialCircuit(cls.Udag_fun, static_args={**static_args}, dynamic_args={**dynamic_args})
                   )
        outcomes = vmap(instance.sample_circuit, in_axes=(0,None))(indices, rho)

        return instance.replace(outcomes=outcomes)

    def predict(self, obs: Observable)->Array:    
        preds = self.compute_preds(obs)
        return self.estimator(preds)

    def compute_preds(self, obs: Observable)->Array:
        preds = vmap(self.prop_inverse_measurement, 
                     in_axes=(0,0,None))(self.outcomes, 
                                         jnp.arange(self.N),
                                         obs)

        return preds

    # shortcut to give U additional arguments
    @staticmethod
    def get_U_params(**kwargs):
        return {}, {}

    @staticmethod
    @abstractmethod
    def U_fun(ind: Array, **kwargs)->None:
        pass       

    # default is simply using a random int
    @classmethod
    def sample_indices(cls, key: Array, shape: tuple[int,...], sample_idx_range: Array,
                       *args, **kwargs)->Array:
        
        indices = random.randint(key, shape, *sample_idx_range)
        return indices

    # returns the properties of the inverse measurements as np array
    # TODO: Write jax.jit for all child classes
    @abstractmethod
    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable)->Array:
        pass


    def sample_circuit(self, ind: Array, state: State):

        @partial(qjit, autograph=True)
        @qp.set_shots(1)
        @qp.qnode(qp.device("lightning.qubit", wires=range(state.n)))
        def circuit(self, ind: Array, state: State):
            state()
            self.U(ind)
            return qp.sample(wires=range(state.n))

        return circuit(self, ind, state)[0]
    
    def ground_truth(self, obs):

        @partial(qjit, autograph=True)
        @qp.qnode(qp.device("lightning.qubit"))
        def circuit(obs):
            self.rho()
            return qp.expval(obs)

        return circuit(obs)


# TODO: Throw warning if full_setting and N odd
# TODO: Throw notimplementederror if the observable is not PauliObservable
class SEEQSTShadow(Shadow):
    # block_idx is the index of the block, whose 
    # binary representation correpsonds to the subset U is sampled from
    # Should be a bitstring or so

    # Changed full_setting not to determine the shape of the members. 
    # Might need to introduce an additional variable if the compiler complains
    full_setting: bool = struct.field(default=False)

    # First n bits of self.indices are for the block encoding, the last one is for the setting
    # redefine this to avoid an extra pass of n
    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               sample_idx_range: Array=jnp.array([0,2]), 
               estimator: Estimator=Estimator(), 
               full_setting: bool=False):
        return super().create(key, N, rho, sample_idx_range, estimator, n=rho.n, full_setting=full_setting)

    @classmethod
    def sample_indices(cls, key: Array, shape: tuple[int,...], sample_idx_range, 
                       n: int, full_setting: bool)->Array:
        N = shape[0]

        def full_fn(key: Array, indices: Array):
            N2 = N//2
            init_indices = random.randint(key, (N2,n), *sample_idx_range)
            indices = indices.at[:N2,:n].set(init_indices)
            indices = indices.at[N2:2*N2,:n].set(init_indices)
            indices = indices.at[:N2,n].set(jnp.zeros((init_indices.shape[0],), dtype=indices.dtype))
            indices = indices.at[N2:2*N2,n].set(jnp.ones((init_indices.shape[0],), dtype=indices.dtype))
            return indices

        def rand_fn(key: Array, indices: Array):
            key1, key2 = random.split(key)
            init_indices = super(SEEQSTShadow, cls).sample_indices(key1, (N,n), sample_idx_range)
            setting_indices = random.randint(key2, (N,), 0, 2)
            indices = indices.at[:,:n].set(init_indices)
            indices = indices.at[:,n].set(setting_indices)

            return indices

        indices = jnp.zeros((N,n+1), dtype=int)

        return cond(full_setting, full_fn, rand_fn, key, indices)

        # Convert circuit text to JAX arrays
        # Selective circuit texts - can be modified to improve efficiency
        # circuits = flatten_list(sel_circ_text) 
        # If want to vectorize, need to make the circuits same length and make the circuit function depend on parameters
    @staticmethod
    def U_fun(ind: Array, **kwargs)->None:
        n = ind.shape[0] - 1
        block_idx, xy = ind[:n], ind[n]
        build_parallel_entangler_blocks(block_idx,n, xy)

    @staticmethod
    def Udag_fun(ind: Array, **kwargs)->None:
        n = ind.shape[0] - 1
        block_idx, xy = ind[:n], ind[n]
        build_parallel_entangler_blocks(block_idx,n,xy,reversed=True)

    # Using b_i as binary representation of 0 <= i < 2**n
    # formula: 2(\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>) - Id/2**n \
    # + 2**(n+1) ( \rho - (\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>)).
    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
            raise NotImplementedError()

        @partial(qjit, autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
        def circuit_rho(outcome, U, obs, index):
            post_meas_state_gates(outcome)
            U(index)
            obs.circuit()
            return qp.expval(obs.op())     
        exp = cond(obs.is_ZType, lambda: 0, lambda: self.n)   
        estimate = 2**(exp+1)*circuit_rho(outcome, self.Udag, obs, self.indices[ind]) 

        return estimate
    
class PauliShadow(Shadow):

    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               estimator: Estimator = Estimator(), *args, **kwargs):
        return super().create(key, N, rho, jnp.array([0,3]), estimator, *args, **kwargs)

    @staticmethod
    def U_fun(ind: Array, **kwargs):
        n = ind.shape[0]
        for i in range(n):
            if ind[i] == 0:
                qp.Hadamard(i)
            elif ind[i] == 1:
                HadjS()(i)

    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
            raise NotImplementedError()
        
        @partial(qjit, autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
        def circuit_rho(outcome, U, obs, index):
            post_meas_state_gates(outcome)
            U(index)
            # here we do not have to apply the obs circuit, because the shadow state is a product state --> trick not needed
            return [qp.expval(oi) for oi in obs.qubit_wise_obs()]
        
        rho = circuit_rho(outcome, self.Udag, obs, self.indices[ind])
        estimate = jnp.prod(3*jnp.array(rho) - jnp.real(jnp.array(obs.qubit_wise_trace())))

        return estimate        

class CliffordShadow(Shadow):
    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               estimator: Estimator = Estimator(), *args, **kwargs):
        return super().create(key, N, rho, jnp.array([]), estimator, *args, **kwargs)

    # Change to sampling the tableaus
    @classmethod
    def sample_indices(cls, key: Array, shape: tuple[int,...], sample_idx_range: Array, 
                       n: int, *args, **kwargs) -> Array:
        # indices are stacked matrices of tableau
        N = shape[0]
        key1, key2 = random.split(key)
        pauli_array = super(CliffordShadow, cls).sample_indices(key1, (N,n), jnp.array([0,3]))
        keys = random.split(key2, N)
        out_mats = vmap(create_tableau, in_axes=(0,None))(keys, n)
        gammadelta = out_mats[:4]
        hS = tuple([mat[:,:,None] for mat in out_mats[4:]])

        return jnp.concat(gammadelta+hS+(pauli_array[:,:,None],), axis=-1)

    @staticmethod
    def U_fun(ind: Array, **kwargs):
        n = ind.shape[0]
        gammadelta = ind[:4*n]
        hSO = [vec[0] for vec in gammadelta[4*n:]]
        canonical_form(*gammadelta,*hSO)

    def prop_inverse_measurement(self, outcome, ind, obs: Observable) -> Array:
        @partial(qjit, autograph=True)
        @qp.qnode(qp.device("default.qubit", wires=range(self.n)))
        def circuit_rho(outcome, U, obs, index):
            post_meas_state_gates(outcome)
            U(index)
            obs.circuit()
            return qp.expval(obs.op())

        rho = circuit_rho(outcome, self.Udag, obs, self.indices[ind])
        estimate = (2**self.n + 1)*rho - jnp.real(obs.trace())

        return estimate

############## TODO: adapt tests #############
def test_seeqst_antidiagonal():
    obs = PauliObservable("XYYY")
    state = HRState(4)
    shadow = SEEQSTShadow(state, [0,2**4])
    for i in range(16):
        for j in range(2):
            ind = [i,j]
            shadow.indices = np.array([ind])
            for outcome in np.ndindex((2,2,2,2)):
                shadow.prop_inverse_measurement(outcome, 0, [obs], test_mode=True)

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

def test_tools():
    pass

def test_shadow():
    paulis = ["XXYY","ZZII"]
    n = 4
    N = 10
    params = jnp.stack([PauliObservable.get_param(pauli) for pauli in paulis])
    obs = PauliObservable(params, n)
    key = jax.random.PRNGKey(1234)
    state = HRState.sample(key, n)
    shadow = SEEQSTShadow.create(key, N, state, full_setting=False)
    pred = vmap(shadow.predict)(obs)
    print(pred)

def test_clifford_jaxed():
    key = random.PRNGKey(12345)
    N = 10
    n = 5
    print(jax.jit(CliffordShadow.sample_indices, static_argnums=(1,3))(key, (N,), jnp.array([0,2]), n))

################################


if __name__ == "__main__":
    # test_seeqst_antidiagonal()
    # test_pauli_shadow()
    # test_clifford_shadow()
    test_clifford_jaxed()
    

    

