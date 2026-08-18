
from itertools import chain
from typing import Any
import pennylane as qp
import numpy as np
from abc import ABC, abstractmethod
import pennylane as qp
from pennylane.operation import Operator
import jax 
from jax import numpy as jnp
from jax import Array, random, jit, vmap
from typing import Callable
from flax import struct
from jax.lax import cond, fori_loop, scan
from functools import partial

# Credit goes to SEEQST
# Can probably be tuned
# Block should already be a binary array
# n is a static argument
def build_parallel_entangler_blocks(selective_block: Array, n: int, xy_ind: Array,
                                    reversed=False):
    """
    Build parallel GHZ-style entangling gate sequences for A SINGLE selective block.
    """
    # Let selective_blocks contain k elements that are 1.
    # The indices in the original array of the first k elements 
    # of the sorted array are 1 and act as active indices
    sorted_indices = jnp.argsort(selective_block, descending=True) 
    # The first k elements of the sorted array are 1 and act as active elements
    sorted_vals = selective_block[sorted_indices] 
    
    # second predicate ensures that only the 'zero' setting is sampled for the Z-type block
    # RXY = qp.RX if xy_ind == 0 and not sorted_vals[0] == 0 else qp.RY
    range_args = cond(reversed, lambda n: (n-2,-1,-1),lambda n: (0,n-1,1), n)
    for i in range(*range_args):
        if sorted_vals[i] == 1:
            # Here, this is equivalent to acting on the active qubits
            # Step 1: Initial rotation on first qubit (arbitrary choice)
            if i == 0:
                if xy_ind == 0 and not sorted_vals[0] == 0:
                    qp.RX(jnp.pi/2, sorted_indices[i])
                else:
                    qp.RY(jnp.pi/2, sorted_indices[i])

            # use current active qubit as head and next one as target
            if sorted_vals[i+1] == 1:
                qp.CNOT(jnp.stack([sorted_indices[i], sorted_indices[i+1]]))

def post_meas_state_gates(outcome):
    for i, oc in enumerate(outcome):
        if oc == 1:
            qp.X(i)

@struct.dataclass
class HadjS:
    def __init__(self) -> None:
        pass

    def __call__(self, i: int):
        return qp.prod(qp.Hadamard(i), qp.adjoint(qp.S)(i)) # type: ignore[reportCallIssue]

@struct.dataclass
class PartialCircuit:
    circuit_fun: Callable = struct.field(pytree_node=False)
    # TODO: Change this when making it shape independent
    static_args: dict = struct.field(pytree_node=False)
    dynamic_args: dict 

    def __call__(self, ind) -> Any:
        self.circuit_fun(ind, **self.static_args, **self.dynamic_args)

    

########### For testing, in final implementation, should use Pennylane too #############

def double_pauli(idx: int, pauli: Callable):   
    return pauli(idx) @ pauli(idx+1) 

def HChain(J: Array) -> Operator:
    paulis = [qp.PauliX, qp.PauliY, qp.PauliZ]
    return sum([Ji * sum([double_pauli(i, P) for P in paulis])
                for i, Ji in enumerate(J)])

###############################

################## Random clifford ###################
# Everything is according to https://arxiv.org/pdf/2003.09412
# Returns h, S from Algorithm 1 in the paper
def sample_mallow(key: Array, n: int):
    A = jnp.arange(1, n+1) # to remove elements, make them 0
    # h = jnp.zeros((n,))
    m = n
    keys = random.split(key, n)

    # maybe have to init hxk on beforehand
    def body_fun(carry: tuple[Array, int], x):
        key = x
        Ai, m = carry
        n = Ai.shape[0]
        # encode h, k pairs into array
        elems = jnp.arange(2*n)
        hs = jnp.tile(jnp.array([[0],[1]]), (1,n))
        ks = jnp.tile(jnp.arange(1,n+1), (2,1))
        mask = jnp.where(jnp.tile(jnp.arange(n), (2,1)) < m, 1, 0)
        prob_vec = 2**(m-1+hs+(m-ks)*(-1)**(1+hs))/(4**m-1)
        prob_vec = prob_vec * mask
        sample = random.choice(key, elems, p=prob_vec.flatten()) 
        hi = sample // n
        ki = sample % n
        # pick ki-th largest element
        maxima = jnp.sort(Ai, stable=False, descending=True)
        j = maxima[ki]
        Ai = Ai.at[j-1].set(0)

        return (Ai, m-1), jnp.array([hi, j])

    _, y = scan(body_fun, init=(A,m), xs=keys)

    return y[:,0], y[:,1]

# PRE: Assume that Gamma and Delta have a random Bernoulli value
# POST: If respective conditions from Algorithm 2 are NOT satisfied, set to 0
@jit
def cond_Gamma(hi: Array, hj: Array, Si: Array, Sj: Array, 
               Gamma_ij: Array):
    
    return cond((hi == 1) & (hj == 1)  \
             | (hi == 1) & (hj == 0) & (Si < Sj) \
             | (hi == 0) & (hj == 1) & (Si > Sj),
             lambda: Gamma_ij, lambda: 0)

@jit
def cond_Delta(hi: Array, hj: Array, Si: Array, Sj: Array, 
               Delta_ij: Array):
    
    return cond((hi == 0) & (hj == 1)  \
             | (hi == 1) & (hj == 1) & (Si > Sj) \
             | (hi == 0) & (hj == 0) & (Si < Sj),
             lambda: Delta_ij, lambda: 0)

# samples the matrices according to Algorithm 2 in the paper
def create_tableau(key: Array, n: int):
    keys = random.split(key, 5)
    Delta, Deltad, Gamma, Gammad = [random.randint(key, (n,n), 0, 2)
                                    for key in keys[:4]]
    h, S = sample_mallow(keys[4], n)

    # This condition also covers the diagonal of Gamma correctly
    Gamma = vmap(vmap(cond_Gamma, 
                      in_axes=(0,None,0,None,0)),
                      in_axes=(None,0,None,0,1))(h,h,S,S,Gamma)
    
    # Gamma, Gammad are symmetric
    Gamma = jnp.tril(Gamma) + jnp.tril(Gamma, k=-1).T
    Gammad = jnp.tril(Gammad) + jnp.tril(Gammad, k=-1).T

    Delta = vmap(vmap(cond_Delta, 
                      in_axes=(0,None,0,None,0)),
                      in_axes=(None,0,None,0,1))(h,h,S,S,Delta)
    
    # Delta, Deltad are lower triangle and 1 on the diagonal
    Delta = jnp.tril(Delta, k=-1) + jnp.eye(n)
    Deltad = jnp.tril(Deltad, k=-1) + jnp.eye(n)
    
    return Gamma, Gammad, Delta, Deltad, h, S

# pennylane circuit operators of F as defined in (2) in the paper 
# note that it is prepared in 'reverse' order so that the correpsondence holds
def F(pauli_indices: Array, Gamma: Array, Delta: Array)->None:
    n = Gamma.shape[0]
    # Delta is lower triangular
    for i in reversed(range(n)):
        for j in reversed(range(i)):
            if Delta[i,j] == 1:
                qp.CNOT(wires=jnp.array([i,j]))

    # Gamma is symmetric
    for i in reversed(range(n)):
        for j in reversed(range(i)):
            if Gamma[i,j] == 1:
                qp.CZ(wires=jnp.array([i,j]))

    for i in range(n):
        if pauli_indices == 1:
            qp.X(i)
        elif pauli_indices == 2:
            qp.Y(i)
        elif pauli_indices == 3:
            qp.Z(i)

        if Gamma[i,i] == 1:
            # in the paper, they call the S gate P
            qp.S(wires=i)
    

# maps canonical form in (3) in the paper to pennylane gates
# note that it is prepared in 'reverse' order so that the correpsondence holds
def canonical_form(Gamma: Array, Delta: Array, 
                   Gammad: Array, Deltad: Array, 
                   h: Array, S: Array,
                   pauli_indices: Array):
    n = Gamma.shape[0]
    F(pauli_indices, Gammad, Deltad)
    qp.Permute(S, n)

    for i in range(n):
        if h == 1:
            qp.H(i)

    F(jnp.zeros((n,), dtype=int), Gamma, Delta)

###########################################

def test_jit():
    @partial(jax.jit, static_argnums=(1,))
    def f(key, n):
        return sample_mallow(key, n)

    key = jax.random.PRNGKey(12345)
    n = 10
    # print(f(key, n))
    # Test passed: no repeating S(i), hi \in {0,1}, probabilities sum up to 1 in each iteration

    @partial(jax.jit, static_argnums=(1,))
    def g(key, n):
        return create_tableau(key, n)

    for mat in g(key, n):
        print(mat)



if __name__ == "__main__":
    test_jit()