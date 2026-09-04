from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, ClassVar, Callable

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxed.tools import utils
from jaxed.tools.utils import build_parallel_entangler_blocks, build_parallel_entangler_blocks_rev, \
                              post_meas_state_gates, HadjS, PartialCircuit, \
                              create_tableau, canonical_form,  \
                              permutation_to_swaps

from jaxed.tools.observable import Observable, PauliObservable, QP_OBS_LIST
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
from jax import Array, random, jit
from flax import struct
from jax import lax
from jax.lax import cond, fori_loop
from functools import partial, lru_cache
import time
import catalyst
from jaxed.tools import clifford
from jaxed.tools.clifford import Tableau, GHZ_type_state_clifford_rev
from jaxed.tools.distributions import uniform


large_width = 400
np.set_printoptions(linewidth=large_width)


# To vectorize, can have an enumeration of relevant objects (including identity if equal length is needed)
# and broadcast over vector with indices
# Add key to this class
class Shadow(ABC, struct.PyTreeNode):
    # n: number of qubits
    # sample_idx_range: indices the U(.) method can take. Can also give as a (multidim-) range [min, max] (latter is default)
    estimator: Estimator
    sample_idx_range: Array = struct.field(pytree_node=False)
    indices: Array
    outcomes: Array
    U: PartialCircuit
    Udag: PartialCircuit
    snapshots: Tableau | None

    # create shadow from N state samples
    @classmethod
    def init(cls, key: Array, n: int, N: int, N_state_reps: int=1, 
                sample_idx_range: Array=jnp.array([]), 
               estimator: Estimator=Estimator(),
               *args, **kwargs):

        keys = random.split(key, N_state_reps)
        in_axes = tuple([0]+[None]*(2+len(args)+len(kwargs)))
        indices = jax.vmap(cls.sample_indices, in_axes=in_axes)(keys, (N,n), 
                                                                sample_idx_range,
                                                                *args, **kwargs)

        kwargs.pop('n', None)
        kwargs.pop('N', None)

        static_args, dynamic_args = cls.get_U_params(N=N, n=n, **kwargs)

        if not hasattr(cls, "Udag_fun"):
            cls.Udag_fun = qp.adjoint(cls.U_fun)

        shape = (N, n) if N_state_reps == 1 else (N_state_reps, N, n)
        instance = cls(estimator=estimator, 
                   sample_idx_range=sample_idx_range, 
                   indices=indices, outcomes=jnp.empty(shape),
                   U=PartialCircuit(cls.U_fun, static_args={**static_args}, dynamic_args={**dynamic_args}),
                   Udag=PartialCircuit(cls.Udag_fun, static_args={**static_args}, dynamic_args={**dynamic_args}),
                   snapshots=Tableau.create(n,N,N_state_reps)
                   )
        
        return instance

    # use this to replace the outcomes
    def _sample(self, indices, state):
        U_treedef = jax.tree_util.tree_structure(self.U)
        state_treedef = jax.tree_util.tree_structure(state)
        circuit = self._sample_circuit(self.n, U_treedef, state_treedef)
        outcomes = circuit(indices, self.U, state)[:,0]

        return outcomes

    def sample(self, states):
        create_samples = lambda shadow, indices, state: shadow._sample(indices, state)
        outcomes = jax.vmap(create_samples, in_axes=(None, 0,0))(self, self.indices, states)
        return self.replace(outcomes=outcomes)

    def create_snapshots(self)->Shadow:
        snapshots = jax.vmap(jax.vmap(self.inverse_circuit_clifford))(self.outcomes,
                                                                     self.indices)

        return self.replace(snapshots=snapshots)

    # Assuming that tableaus has batch dim N
    def estimate_property(self, snapshots: Array, obs: Observable, Ns: Array=jnp.array([0])):
        est_prop = lambda tableau, obs: self._inverse_channel(self.n, 
                                                              tableau.expval(obs.params), 
                                                              obs)
        props = jax.vmap(est_prop,
                        in_axes=(0,None))(snapshots,
                                          obs)
        pred_fun = lambda estimator, props, N: estimator(props, N)
        return jax.vmap(pred_fun, 
                        in_axes=(None, None, 0))(self.estimator, props, Ns)

    def estimate_properties(self, obs: Observable, Ns: Array=jnp.array([0]),
                            batch_size: int | None=None)->Array:

        est_prop = lambda shadow, snapshots, obs: shadow.estimate_property(snapshots, obs, Ns)
        est_props = lambda snapshots: jax.vmap(est_prop,
                                               in_axes=(None, None,0))(self, 
                                                                       snapshots, 
                                                                       obs)
        props = lax.map(est_props, self.snapshots, batch_size=batch_size)

        return props
    

    def predict(self, outcomes: Array, indices: Array, obs: Observable, Ns: Array=jnp.array([0]))->Array:
        U_treedef = jax.tree_util.tree_structure(self.Udag)
        obs_treedef = jax.tree_util.tree_structure(obs)
        inv_circuit = self._inverse_circuit(self.n, 
                                                len(outcomes.shape)-2,
                                                obs_treedef,
                                                U_treedef)

        inv_ocs = inv_circuit(outcomes, self.Udag, obs, indices)
        props = jax.vmap(self._inverse_channel, in_axes=(None,0,None))(self.n, inv_ocs, obs)
        pred_fun = lambda estimator, props, N: estimator(props, N)

        return jax.vmap(pred_fun, 
                        in_axes=(None, None, 0))(self.estimator, props, Ns)

    @classmethod
    @abstractmethod
    def inverse_circuit_clifford(cls, outcome: Array, ind: Array) -> Tableau:
        pass

    # shortcut to give U additional arguments
    @staticmethod
    def get_U_params(**kwargs):
        return {}, {}

    @staticmethod
    @abstractmethod
    def U_fun(ind: Array, **kwargs)->None:
        pass       

    @property
    def N_state_reps(self)->int:
        return self.outcomes.shape[-3]

    @property
    def n(self)->int:
        return self.outcomes.shape[-1]
    
    @property
    def N(self)->int:
        return self.outcomes.shape[-2]
    
    # default is simply using a random int
    @classmethod
    def sample_indices(cls, key: Array, shape: tuple[int,...], sample_idx_range: Array,
                       *args, **kwargs)->Array:
        
        indices = random.randint(key, shape, *sample_idx_range)
        return indices

    # returns the properties of the inverse measurements as np array
    # TODO: Write jax.jit for all child classes
    @classmethod
    @abstractmethod
    def _inverse_channel(cls, n: int, inv_oc: Array, obs: Observable)->Array:
        pass

    @classmethod
    @abstractmethod
    def _inverse_circuit(cls, n: int, ind_axis, obs_treedef, U_treedef) -> Callable[..., Any]:
        pass

    @staticmethod
    @lru_cache(None)
    def _sample_circuit(n: int, U_treedef, state_treedef):
        U_axes = U_treedef.unflatten(
            [None] * U_treedef.num_leaves
        )
        state_axes = state_treedef.unflatten(
            [None] * state_treedef.num_leaves
        )

        @qp.set_shots(1)
        @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
        def circuit(ind, U, state):
            state()
            U(ind)
            return qp.sample()

        return qjit(autograph=True)(catalyst.vmap(circuit, in_axes=(0,U_axes,state_axes)))

    # TODO: Move these functions to utils
    ###################################
    @staticmethod
    @lru_cache(None)
    def _gt_circuit(n: int, state_treedef, obs_treedef):
        obs_axes = obs_treedef.unflatten(
                    [None] * state_treedef.num_leaves
                )
        state_axes = state_treedef.unflatten(
            [0] * state_treedef.num_leaves
        )

        @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
        def circuit(state, obs):
            state()
            obs.circuit()
            return qp.expval(obs.op())
        
        return qjit(autograph=True)(catalyst.vmap(circuit, in_axes=(state_axes, obs_axes)))

    def _ground_truth(self, states: State, obs: Observable):
        state_treedef = jax.tree_util.tree_structure(states)
        obs_treedef = jax.tree_util.tree_structure(obs)
        circuit = self._gt_circuit(self.n, state_treedef, obs_treedef)
        return circuit(states, obs)

    def ground_truth(self, states: State, observables: Observable):
        return jax.vmap(self._ground_truth, in_axes=(None,0), out_axes=1)(states, observables)

    ####################################

# TODO: Throw warning if full_setting and N odd
# TODO: Throw notimplementederror if the observable is not PauliObservable
class SEEQSTShadow(Shadow):
    # block_idx is the index of the block, whose 
    # binary representation correpsonds to the subset U is sampled from
    # Should be a bitstring or so

    # Changed full_setting not to determine the shape of the members. 
    # Might need to introduce an additional variable if the compiler complains
    full_setting: bool = struct.field(pytree_node=False,
                                      default=False)

    # First n bits of self.indices are for the block encoding, the last one is for the setting
    # redefine this to avoid an extra pass of n
    @classmethod
    def init(cls, key: Array, n: int, N: int, N_state_reps: int, 
               sample_idx_range: Array=jnp.array([0,2]), 
               estimator: Estimator=Estimator(), 
               distribution: Callable=uniform,
               *args, **kwargs):
        return super().init(key, n, N, N_state_reps, sample_idx_range, estimator, distribution)

    # Distribution has to be a function matching the pattern below and return SEEQST binary indices
    # of shape (N, n+1)
    @classmethod
    def sample_indices(cls, key: Array, shape: tuple[int,...], sample_idx_range, 
                       distribution: Callable, *args, **kwargs)->Array:
        N, n = shape
        indices = distribution(key, N, n, sample_idx_range, *args, **kwargs)
    
        return indices

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
        build_parallel_entangler_blocks_rev(block_idx,n,xy)

    @classmethod
    @lru_cache(None)
    def _inverse_circuit(cls, n: int, ind_axis, obs_treedef, U_treedef) -> Callable[..., Any]:
        obs_axes = obs_treedef.unflatten(
            [None] * obs_treedef.num_leaves
        )

        U_axes = U_treedef.unflatten(
                    [None] * U_treedef.num_leaves
                )

        @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
        def circuit_rho(outcome, U, obs, index):
            post_meas_state_gates(outcome)
            U(index)
            obs.circuit()
            return qp.expval(obs.op())  
        
        circuits = catalyst.vmap(circuit_rho, in_axes=(0, U_axes, obs_axes, ind_axis))

        return qjit(autograph=True)(circuits)

    @classmethod
    def inverse_circuit_clifford(cls, outcome: Array, ind: Array) -> Tableau:
        n = outcome.shape[-1]
        selective_block, xy = ind[:n], ind[n]
        tableau = Tableau.create(n)
        tableau = tableau.MultiPauli(outcome)
        tableau = GHZ_type_state_clifford_rev(selective_block, xy, tableau)
        return tableau

    @classmethod
    def _inverse_channel(cls, n: int, inv_oc: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
                    raise NotImplementedError()
        exp = cond(obs.is_ZType, lambda: 0, lambda: n)   
        estimate = 2**(exp+1)*inv_oc
        return estimate
    # Using b_i as binary representation of 0 <= i < 2**n
    # formula: 2(\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>) - Id/2**n \
    # + 2**(n+1) ( \rho - (\sum_{b_i} |b_i><b_i| <b_i|\rho|b_i>)).
    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
            raise NotImplementedError()

        @qjit(autograph=True)
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
    def init(cls, key: Array, n: int, N: int, N_state_reps:int, sample_idx_range: Array=jnp.array([0,3]),
               estimator: Estimator = Estimator(), *args, **kwargs):
        return super().init(key, n, N, N_state_reps, jnp.array([0,3]), estimator, *args, **kwargs)

    @staticmethod
    def U_fun(ind: Array, **kwargs):
        n = ind.shape[0]
        for i in range(n):
            if ind[i] == 0:
                qp.Hadamard(i)
            elif ind[i] == 1:
                qp.adjoint(qp.S)(i)
                qp.Hadamard(i)

    @staticmethod
    def Udag_fun(ind: Array, **kwargs):
        n = ind.shape[0]
        
        for i in range(n):
            if ind[i] == 0:
                qp.Hadamard(i)
            elif ind[i] == 1:
                qp.Hadamard(i)
                qp.S(i)

    @classmethod
    @lru_cache(None)
    def _inverse_circuit_deprecated(cls, n: int, ind_axis, obs_treedef, U_treedef) -> Callable[..., Any]:
        obs_axes = obs_treedef.unflatten(
            [None] * obs_treedef.num_leaves
        )

        U_axes = U_treedef.unflatten(
                    [None] * U_treedef.num_leaves
                )
        # TODO: Implement this one manually, since it is a product state
        @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
        def circuit_rho(outcome, U, obs, index):
            post_meas_state_gates(outcome)
            U(index)
            # here we do not have to apply the obs circuit, because the shadow state is a product state --> trick not needed
            return [qp.expval(oi) for oi in obs.qubit_wise_obs()]

        circuits = qjit(autograph=True)(catalyst.vmap(circuit_rho, in_axes=(0, U_axes, obs_axes, ind_axis)))
        return circuits

    @classmethod
    def _inverse_circuit(cls, n: int, ind_axis, obs_treedef, U_treedef) -> Callable[..., Any]:
        paulis = jnp.stack([op(0).matrix() for op in QP_OBS_LIST])
        Us = jnp.stack([qp.matrix(op(0)) for op in [qp.Hadamard, lambda i: qp.adjoint(HadjS(i)), qp.Identity]])
        def single_qubit_expval(qubit_oc, U, obs, index):
            state = jnp.zeros((2,), dtype=complex).at[qubit_oc].set(1)
            state = Us[index] @ state
            return jnp.real(jnp.vdot(state, paulis[obs.params] @ state))

        circuits = jax.vmap(jax.vmap(single_qubit_expval, 
                                     in_axes=(0, None, 0, ind_axis)), 
                                     in_axes=(0, None, None, ind_axis))

        return circuits 

    # do nothing for compatibility
    def create_snapshots(self) -> Shadow:
        return self

    def estimate_properties(self, obs: Observable, Ns: Array=jnp.array([0]),
                                batch_size: int | None=None)->Array:
        est_prop = lambda shadow, outcomes, indices, obs: shadow.predict(outcomes, indices, obs, Ns)
        def est_props(ocind):
            outcomes, indices = ocind
            return jax.vmap(est_prop,
                            in_axes=(None, None, None,0,))(self, 
                                                    outcomes,
                                                    indices, 
                                                    obs)
        
        props = lax.map(est_props, (self.outcomes, self.indices), batch_size=batch_size)
        return props

    @classmethod
    def inverse_circuit_clifford(cls, outcome: Array, ind: Array) -> Tableau:
        raise NotImplementedError("Use the more efficient _inverse_circuit.")

    @classmethod
    def _inverse_channel(cls, n: int, inv_oc: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
                    raise NotImplementedError()

        estimate = jnp.prod(3*inv_oc - jnp.real(jnp.array(obs.qubit_wise_trace())))
        return estimate

    def prop_inverse_measurement(self, outcome: Array, ind: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
            raise NotImplementedError()

        @qjit(autograph=True)
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

    # Change to sampling the tableaus
    @classmethod
    def sample_indices(cls, key: Array, shape: tuple[int,...], sample_idx_range: Array,
                       *args, **kwargs) -> Array:
        # indices are stacked matrices of tableau
        N, n = shape
        key1, key2 = random.split(key)
        pauli_array = super(CliffordShadow, cls).sample_indices(key1, 
                                                                (N,n), 
                                                                jnp.array([0,3]))
        keys = random.split(key2, N)
        out_mats = jax.vmap(create_tableau, in_axes=(0,None))(keys, n)
        gammadelta = out_mats[:4]

        swap_inds = jax.vmap(permutation_to_swaps, in_axes=0)(out_mats[5])
        
        return jnp.concat(gammadelta+(out_mats[4][:,:,None],) \
                          + (pauli_array[:,:,None],) \
                          + (swap_inds,) \
                          + (out_mats[5][:,:,None],), # S
                          axis=-1)

    @staticmethod
    def U_fun(ind: Array, **kwargs):
        n = ind.shape[0]
        gammadelta = [ind[:,i*n:(i+1)*n] for i in range(4)]
        hO = [ind[:,4*n+i] for i in range(2)]
        canonical_form(*gammadelta,*hO,swap_indices=ind[:,4*n+2:4*n+4])

    @staticmethod
    def Udag_fun(ind: Array, **kwargs):
        n = ind.shape[0]
        gammadelta = [ind[:,i*n:(i+1)*n] for i in range(4)]
        hO = [ind[:,4*n+i] for i in range(2)]
        utils.canonical_form_rev(*gammadelta,*hO,swap_indices=ind[:,4*n+2:4*n+4])

    @classmethod
    @lru_cache(None)
    def _inverse_circuit(cls, n: int, ind_axis, obs_treedef, U_treedef) -> Callable[..., Any]:
        obs_axes = obs_treedef.unflatten(
            [None] * obs_treedef.num_leaves
        )

        U_axes = U_treedef.unflatten(
                    [None] * U_treedef.num_leaves
                )

        @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
        def circuit_rho(outcome, U, obs, index):
            post_meas_state_gates(outcome)
            U(index)
            obs.circuit()
            return qp.expval(obs.op())  

        circuits = catalyst.vmap(circuit_rho, in_axes=(0, U_axes, obs_axes, ind_axis))

        return qjit(autograph=True)(circuits)

    @classmethod
    def inverse_circuit_clifford(cls, outcome: Array, ind: Array) -> Tableau:
        n = outcome.shape[-1]
        tableau = Tableau.create(n)
        tableau = tableau.MultiPauli(outcome)
        gammadelta = [ind[:,i*n:(i+1)*n] for i in range(4)]
        hO = [ind[:,4*n+i] for i in range(2)]
        S = ind[:,4*n+4]
        tableau = clifford.canonical_form_rev(tableau, *gammadelta, *hO, S)

        return tableau


    @classmethod
    def _inverse_channel(cls, n: int, inv_oc: Array, obs: Observable) -> Array:
        if not isinstance(obs, PauliObservable):
                    raise NotImplementedError()
  
        estimate = (2**n + 1)*inv_oc - jnp.real(obs.trace())
        return estimate

    def prop_inverse_measurement(self, outcome, ind, obs: Observable) -> Array:
        @qjit(autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=range(self.n)))
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

def test_seeqst_shadow():
    paulis = ["XXYY","ZZII"]
    n = 10
    N = 1000
    N_state_reps = 100
    params = jnp.stack([PauliObservable.get_param(pauli) for pauli in paulis])
    obs = PauliObservable.init(paulis)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    start = time.time()
    shadow = SEEQSTShadow.init(key, n, N, N_state_reps)
    outcomes_fun = jax.vmap(lambda shadow, state: shadow.sample(state), in_axes=(0,0))
    print("started timing")
    print(shadow.indices.shape, states.state_dm.shape)
    outcomes = outcomes_fun(shadow,states)
    shadow = shadow.replace(outcomes=outcomes)
    end = time.time()
    print(end-start)
    print("sampled")
    # shadow.prop_inverse_measurement(jnp.ones(n,), 0, obs)
    start = time.time()
    pred_fun = lambda shadow, obs: shadow.predict(obs)
    pred_fun = jax.vmap(jax.vmap(pred_fun, in_axes=(None, 0)), in_axes=(0,None))
    print(pred_fun(shadow, obs))
    end = time.time()
    print(end-start)

def test_clifford_jaxed():
    n = 10
    paulis = ["X"*n,"Y"*n]*4
    N = 10000
    N_state_reps = 500
    params = jnp.stack([PauliObservable.get_param(pauli) for pauli in paulis])
    obs = PauliObservable.init(paulis)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    start = time.time()
    shadow = CliffordShadow.init(key, n, N, N_state_reps)
    outcomes_fun = jax.vmap(lambda shadow, state: shadow.sample(state), in_axes=(0,0))
    print("started timing")
    print(shadow.indices.shape, states.state_dm.shape)
    outcomes = outcomes_fun(shadow,states)
    shadow = shadow.replace(outcomes=outcomes)
    end = time.time()
    print(end-start)
    print("sampled")
    # shadow.prop_inverse_measurement(jnp.ones(n,), 0, obs)
    start = time.time()
    pred_fun = lambda shadow, obs: shadow.predict(obs)
    pred_fun = jit(jax.vmap(jax.vmap(pred_fun, in_axes=(None, 0)), in_axes=(0,None)))
    print(pred_fun(shadow, obs))
    end = time.time()
    print(end-start)

def test_pauli_jaxed():
    n = 10
    paulis = ["X"*n,"Y"*n]
    N = 100000
    N_state_reps = 3
    params = jnp.stack([PauliObservable.get_param(pauli) for pauli in paulis])
    obs = PauliObservable.init(paulis)
    print(obs.params)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    start = time.time()
    shadow = PauliShadow.init(key, n, N, N_state_reps)
    outcomes_fun = jax.vmap(lambda shadow, state: shadow.sample(state), in_axes=(0,0))
    print("started timing")
    print(states.state_dm.shape, shadow.indices.shape)
    outcomes = outcomes_fun(shadow,states)
    print(outcomes.shape)
    shadow = shadow.replace(outcomes=outcomes)
    end = time.time()
    print(end-start)
    print("sampled")
    # shadow.prop_inverse_measurement(jnp.ones(n,), 0, obs)
    start = time.time()
    pred_fun = lambda shadow, obs: shadow.predict2(obs)
    pred_fun = jax.vmap(jax.vmap(pred_fun, in_axes=(None, 0)), in_axes=(0,None))
    print(pred_fun(shadow, obs))
    end = time.time()
    print(end-start)
    start = time.time()
    pred_fun = lambda shadow, obs: shadow.predict(obs)
    pred_fun = jax.vmap(jax.vmap(pred_fun, in_axes=(None, 0)), in_axes=(0,None))
    print(pred_fun(shadow, obs))
    end = time.time()
    print(end-start)

def test_clifford_sim():
    n = 3
    paulis = ["X"*n,"Y"*n, "Z"*n]
    N = 100000
    N_state_reps = 5
    obs = PauliObservable.init(paulis)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    print("started timing")
    start = time.time()
    shadow = CliffordShadow.init(key, n, N, N_state_reps)
    shadow = shadow.sample(states)
    end = time.time()
    print(end-start)
    print("sampled")
    start = time.time()
    @jit
    def fun(shadow, obs):
        shadow = jit(shadow.create_snapshots)()
        props = jit(shadow.estimate_properties)(obs)
        return props
    print(shadow.ground_truth(states, obs).T)
    out = fun(shadow, obs)
    print(out)
    end = time.time()
    print(end-start)
    print("started timing again")
    start = time.time()
    pred_fun = lambda shadow, obs: shadow.predict(obs)
    pred_fun = jax.vmap(jax.vmap(pred_fun, in_axes=(None, 0)), in_axes=(0,None))
    print(pred_fun(shadow, obs))
    end = time.time()
    print(end-start)

def test_seeqst_sim():
    n = 3
    paulis = ["X"*n,"Y"*n, "Z"*n]
    N = 10000

    N_state_reps = 5
    obs = PauliObservable.init(paulis)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    print("started timing")
    start = time.time()
    shadow = SEEQSTShadow.init(key, n, N, N_state_reps)
    shadow = shadow.sample(states)
    end = time.time()
    print(end-start)
    print("sampled")
    start = time.time()
    shadow = jit(shadow.create_snapshots)()
    @jit
    def fun(shadow, obs, N):
        props = shadow.estimate_properties(obs, N)
        return props

    Ns = jnp.array([1e2, 1e3, 1e4, -1], dtype=int)
        
    out = jax.vmap(fun, in_axes=(None, None, 0))(shadow, obs, Ns)
    print(out)
    end = time.time()
    print(states.state_dm.shape)
    print(shadow.ground_truth(states, obs))
    print(end-start)
    """print("started timing again")
    start = time.time()
    pred_fun = lambda shadow, obs: shadow.predict(obs)
    pred_fun = jax.vmap(jax.vmap(pred_fun, in_axes=(None, 0)), in_axes=(0,None))
    print(pred_fun(shadow, obs))
    end = time.time()
    print(end-start)"""

def test_pauli_new():
    n = 3
    paulis = ["X"*n,"Y"*n, "Z"*n]
    N = 10000
    N_state_reps = 5
    obs = PauliObservable.init(paulis)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    print("started timing")
    start = time.time()
    shadow = PauliShadow.init(key, n, N, N_state_reps)
    shadow = shadow.sample(states)
    end = time.time()
    print(end-start)
    print("sampled")
    start = time.time()
    @jit
    def fun(shadow, obs):
        shadow = jit(shadow.create_snapshots)()
        props = jit(shadow.estimate_properties)(obs)
        return props
        
    out = fun(shadow, obs)
    print(out)
    end = time.time()
    print(states.state_dm.shape)
    print(shadow.ground_truth(states, obs).T)
    print(end-start)

################################


if __name__ == "__main__":
    # test_seeqst_antidiagonal()
    # test_pauli_shadow()
    # test_clifford_shadow()
    # test_clifford_jaxed()
    # test_pauli_jaxed()
    # test_seeqst_shadow()
    # test_clifford_sim()
    test_seeqst_sim()
    # test_pauli_new()
    

    

