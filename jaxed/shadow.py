from abc import ABC, abstractmethod
from typing import List, Any, ClassVar
from jaxed.tools.utils import build_parallel_entangler_blocks, post_meas_state_gates, HadjS, random_clifford
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

large_width = 400
np.set_printoptions(linewidth=large_width)


# To vectorize, can have an enumeration of relevant objects (including identity if equal length is needed)
# and broadcast over vector with indices
# Add key to this class
class Shadow(ABC, struct.PyTreeNode):
    # n: number of qubits
    # gate_indices: indices the U(.) method can take. Can also give as a (multidim-) range [min, max] (latter is default)
    rho: State
    estimator: Estimator
    gate_indices: Array = struct.field(pytree_node=False)
    n: int = struct.field(pytree_node=False)
    N: int = struct.field(pytree_node=False)
    indices: Array
    outcomes: Array

    # create shadow from N state samples
    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               gate_indices: Array=jnp.array([]), 
               estimator: Estimator=Estimator(),
               *args, **kwargs):

        indices = cls.sample_indices(key, N, gate_indices,*args, **kwargs)
        outcomes = vmap(cls.sample_circuit, in_axes=(0,None))(indices, rho)

        return cls(rho=rho, estimator=estimator, 
                   gate_indices=gate_indices, n=rho.n, 
                   N=N, indices=indices, outcomes=outcomes)

    def predict(self, obs: Observable)->Array:    
        preds = self.compute_preds(obs)
        return self.estimator(preds)

    def compute_preds(self, obs: Observable)->Array:
        preds = vmap(self.prop_inverse_measurement, 
                     in_axes=(0,0,None))(self.outcomes, 
                                         jnp.arange(self.N),
                                         obs)

        return preds

    @abstractmethod
    def U(self, ind: Array)->None:
        pass          

    # default is simply using a random int
    @classmethod
    def sample_indices(cls, key: Array, N: int, gate_indices: Array,
                       *args, **kwargs)->Array:
        
        indices = random.randint(key, (N,), *gate_indices)
        return indices

    # returns the properties of the inverse measurements as np array
    # TODO: Write jax.jit for all child classes
    @abstractmethod
    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable)->Array:
        pass

    @classmethod
    def sample_circuit(cls, ind: Array, state: State):
        @qjit
        @qp.set_shots(1)
        @qp.qnode(qp.device("lightning.qubit", wires=range(state.n)))
        def circuit(ind: Array):
            state()
            cls.U(ind)
            return qp.sample(wires=list(range(state.n)))

        return circuit(ind)
    
    def ground_truth(self, obs):

        @qjit
        @qp.qnode(qp.device("lightning.qubit"))
        def circuit(obs):
            self.rho()
            return qp.expval(obs)

        return circuit(obs)
    
    def rho_pm(self, outcome: Array, ind: Array):
        post_meas_state_gates(outcome)
        qp.adjoint(self.U)(self.indices[ind])


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
               gate_indices: Array=jnp.array([0,2]), 
               estimator: Estimator=Estimator(), 
               full_setting: bool=False):
        return super().create(key, N, rho, gate_indices, estimator, n=rho.n, full_setting=full_setting)

    @classmethod
    def sample_indices(cls, key: Array, N: int, gate_indices, 
                       n: int, full_setting: bool)->Array:

        def full_fn(key: Array, indices: Array):
            N2 = N//2
            init_indices = indices = random.randint(key, (N,n), *gate_indices)
            indices = indices.at[:N2,:n].set(init_indices)
            indices = indices.at[N2:2*N2,:n].set(init_indices)
            indices = indices.at[:N2,n].set(jnp.zeros_like(init_indices))
            indices = indices.at[N2:2*N2,n].set(jnp.ones_like(init_indices))
            return indices

        def rand_fn(key: Array, indices: Array):
            key1, key2 = random.split(key)
            init_indices = super(SEEQSTShadow, cls).sample_indices(key1, N, gate_indices)
            setting_indices = random.randint(key2, (N,n), 0, 2)
            indices = indices.at[:,:n].set(init_indices)
            indices = indices.at[:,n].set(setting_indices)

            return indices

        indices = jnp.zeros((N,n+1))

        return cond(full_setting, full_fn, rand_fn, key=key, indices=indices)

        # Convert circuit text to JAX arrays
        # Selective circuit texts - can be modified to improve efficiency
        # circuits = flatten_list(sel_circ_text) 
        # If want to vectorize, need to make the circuits same length and make the circuit function depend on parameters
    def U(self, ind: Array)->None:
        block_idx, xy = ind[:self.n], ind[self.n]
        build_parallel_entangler_blocks(block_idx, self.n, xy)

    # Using b_i as binary representation of 0 <= i < 2**n
    # formula: 2(\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>) - Id/2**n \
    # + 2**(n+1) ( \rho - (\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>)).
    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable) -> Array:

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
        def circuit_rho(outcome):
            post_meas_state_gates(outcome)
            qp.adjoint(self.U)(self.indices[ind])
            obs.circuit()
            return qp.expval(obs.op())
            

        estimate = 2**(self.n+1)*circuit_rho(outcome) 

        return estimate
    
class PauliShadow(Shadow):

    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               estimator: Estimator = Estimator(), *args, **kwargs):
        return super().create(key, N, rho, jnp.array([0,3]), estimator, *args, **kwargs)
        
    def U(self, ind: Array):
        for i in range(self.n):
            if ind[i] == 0:
                qp.Hadamard(i)
            elif ind[i] == 1:
                HadjS()(i)

    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: PauliObservable) -> Array:

        @qjit
        @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
        def circuit_rho(outcome):
            self.rho_pm(outcome, ind)
            # here we do not have to apply the obs circuit, because the shadow state is a product state --> trick not needed
            return [qp.expval(oi) for oi in obs.qubit_wise_obs()]

        estimate = jnp.prod(3*jnp.array(circuit_rho(outcome)) - jnp.real(jnp.array(obs.qubit_wise_trace())))

        return estimate        

class CliffordShadow(Shadow):
    @classmethod
    def create(cls, key: Array, N: int, rho: State, 
               estimator: Estimator = Estimator(), *args, **kwargs):
        return super().create(key, N, rho, jnp.array([]), estimator, *args, **kwargs)

    @classmethod
    def sample_indices(cls, key: Array, N: int, gate_indices: Array, *args, **kwargs) -> Array:
        # indices simply become keys, since the random clifford directly maps a key to a circuit
        return random.split(key, N)

    def U(self, ind: Array):
        random_clifford(self.indices[ind], self.n)

    def prop_inverse_measurement(self, outcome, ind, obs: Observable) -> Array:
        
        @qp.qnode(qp.device("default.qubit", wires=range(self.n)))
        def circuit_rho(outcome):
            self.rho_pm(outcome, ind)
            obs.circuit()
            return qp.expval(obs.op())
    
        estimate = (2**self.n + 1)*circuit_rho(outcome) - jnp.real(obs.trace())

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

################################


if __name__ == "__main__":
    test_seeqst_antidiagonal()
    # test_pauli_shadow()
    # test_clifford_shadow()
    

    

