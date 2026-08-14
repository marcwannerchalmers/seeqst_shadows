
from itertools import chain
from typing import Any
import pennylane as qp
import numpy as np
from abc import ABC, abstractmethod
import pennylane as qp
from pennylane.operation import Operator
import jax 
from jax import numpy as jnp
from jax import Array
from typing import Callable
from flax import struct

# Credit goes to SEEQST
# Can probably be tuned
# Block should already be a binary array
# n is a static argument
def build_parallel_entangler_blocks(selective_block: Array, n: int, xy_ind: Array):
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
    RXY = qp.RX if xy_ind == 0 and not sorted_vals[0] == 0 else qp.RY
    for i in range(n-1):
        if sorted_vals[i] == 1:
            # Here, this is equivalent to acting on the active qubits
            # Step 1: Initial rotation on first qubit (arbitrary choice)
            if i == 0:
                RXY(jnp.pi/2, sorted_indices[i])

            # use current active qubit as head and next one as target
            if sorted_vals[i+1] == 1:
                qp.CNOT(sorted_indices[i], sorted_indices[i+1])

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

########### For testing, in final implementation, should use Pennylane too #############

def double_pauli(idx: int, pauli: Callable):   
    return pauli(idx) @ pauli(idx+1) 

def HChain(J: Array) -> Operator:
    paulis = [qp.PauliX, qp.PauliY, qp.PauliZ]
    return sum([Ji * sum([double_pauli(i, P) for P in paulis])
                for i, Ji in enumerate(J)])

###############################

################## Random clifford ###################
def sample_mallow(key: Array, n: int):
    pass

def create_tableau(key: Array, n: int):
    pass

def random_clifford(key: Array, n: int):
    pass

###########################################

def test_jit():
    pass

if __name__ == "__main__":
    test_jit()