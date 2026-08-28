import pennylane as qp
from catalyst import qjit
from jaxed.tools.clifford import Tableau
from jaxed.tools.utils import build_parallel_entangler_blocks_rev, build_parallel_entangler_blocks, post_meas_state_gates
from jax import numpy as jnp
from jax import Array
from jax.lax import fori_loop
from jaxed.tools.observable import PauliObservable
from jaxed.tools.state import HRState
import jax



def simulate_clifford(outcome: Array, ind: Array, obs: PauliObservable):
    n = outcome.shape[-1]
    selective_block, xy = ind[:n], ind[n]
    tableau = Tableau.create(n)
    tableau = tableau.MultiPauli(outcome)
    # Read utils.parallel_entangler_blocks for more explanation
    sorted_indices = jnp.argsort(selective_block, descending=True) 
    sorted_vals = selective_block[sorted_indices] 

    # TODO: Add this with the reversed argument to clifford
    def body_fun(i, tableau: Tableau):
        ind = n-2-i # reversed order 
        return tableau.CNOT(sorted_indices[ind],
                            sorted_indices[ind+1],
                            (sorted_vals[ind] == 1) & (sorted_vals[ind+1] == 1))

    tableau = fori_loop(0, n-1, body_fun, tableau)

    theta = -jnp.pi/2
    tableau = tableau.PauliRot(jnp.array(0, dtype=int), 
                                theta, 
                                sorted_vals[0]*(xy+1)) # applies nothing if indices are all 0

    return tableau.expval(obs.params)

def simulate_qp(outcome: Array, ind: Array, obs: PauliObservable, circuit):
    n = outcome.shape[-1]
    

    return circuit(outcome, ind, obs)

def test_functions():
    n = 3
    N = 1000
    outcomes = jax.random.randint(jax.random.PRNGKey(12345),
                                  (N,n), 0, 2)
    inds = jax.random.randint(jax.random.PRNGKey(54321),
                                  (N,n+1), 0, 2)

    obs = PauliObservable.init_random(jax.random.PRNGKey(412354),n=n, N=N)
    obs = [obs.replace(params=1*jnp.ones(n,dtype=int)) for i in range(N)]
    dev = qp.device("lightning.qubit", wires=n)
    state = HRState.init_random(jax.random.PRNGKey(1111), 1, n)
    state = state.replace(state_dm=state.state_dm[0])
    # @qjit(autograph=True)
    @qp.qnode(dev)
    def circuit(outcome, ind, obs):
        n = outcome.shape[-1]
        selective_block, xy = ind[:n], ind[n]
        post_meas_state_gates(outcome)
        build_parallel_entangler_blocks(selective_block, n, xy)
        obs.circuit()
        return qp.expval(obs.op())

    

    for i in range(N):
        out1 = simulate_clifford(outcomes[i], inds[i], obs[i])
        out2 = simulate_qp(outcomes[i], inds[i], obs[i], circuit)
        print(i, out1, out2, outcomes[i], inds[i])

if __name__ == "__main__":
    test_functions()
