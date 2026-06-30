from SEEQST.tools.setup import generate_experiment, generate_selective_elements, \
    build_parallel_entangler_blocks, build_non_entangling_circuits, parse_circuit, generate_observable_sets
import qutip as qt
import numpy as np
import jax.numpy as jnp
import jax
from SEEQST.tools.processing import process_data,  flatten_list, parse_circuit_to_qobj, \
    data_predict_from_rho_sampled, prepare_state, density_matrix
import matplotlib.pyplot as plt

X = jnp.array([[0,1],
              [1,0]], dtype=jnp.complex128)

Y = jnp.array([[0,-1j],
              [1j,0]], dtype=jnp.complex128)

Z = jnp.array([[1,0],
              [0,-1]], dtype=jnp.complex128)

I = jnp.array([[1,0],
              [0,1]], dtype=jnp.complex128)

def observable_from_string(obs_string):
    obs_dict = {"I": I, "X": X, "Y": Y, "Z": Z}
    res = jnp.array([1], dtype=jnp.complex128)
    for char in obs_string:
        res = jnp.kron(res, obs_dict[char])

    return res

def all_one_pauli(pauli_char):
    def obs_fun(n_qubits):
        return [pauli_char for _ in range(n_qubits)]
    
    return obs_fun

# need to adapt this and estimate to make it jittable
# note that the indices will always be the same once initialized
def k_local_pauli(key, pauli_char, k):
    def obs_fun(n_qubits):
        indices = jax.random.choice(key, jnp.arange(0, n_qubits), shape=(k,), replace=False,)
        obs_string = ["I" for _ in range(n_qubits)]
        for ind in range(k):
            obs_string[indices[ind]] = pauli_char

    return obs_fun


def shadow_estimate(rho_gt, obs, N_shadow, key):
    shots = 1
    n_qubits = int(jnp.log2(rho_gt.shape[0]))
    # sample N settings, i.e. blocks for shadow
    block_indices = jax.random.randint(key, (N_shadow,),minval=0, maxval=2**n_qubits).tolist()
    # result = generate_selective_elements(block_indices, None, n_qubits)
    # observable_dict = generate_observable_sets(block_indices, n_qubits)
    sel_circ_text = build_parallel_entangler_blocks(block_indices, n_qubits)
    # Convert circuit text to JAX arrays
    # Selective circuit texts - can be modified to improve efficiency
    circuits = flatten_list(sel_circ_text)  # Requires CNOT gates

    # circuits = flatten_list(sel_circ_text_non_entangle) # Uncomment to use non-entangling circuits

    # Convert circuit text to unitary matrices
    unitaries = parse_circuit_to_qobj(circuits, n_qubits)

    # Convert Qobj to JAX numpy arrays
    unitaries_jnp = jnp.array([uni.full() for uni in unitaries])
    # print("constructing gt...")
    rho_gt = qt.rand_dm([[2]*n_qubits])  # Random density matrix
    # print("sample rho...")
    data = (data_predict_from_rho_sampled(jnp.array(rho_gt.full()), unitaries_jnp, shots=shots))
    # print("processing data...")
    rho_hat = process_data(data=data,unitaries_jnp=unitaries_jnp,selective_blocks=block_indices,shots=1,N=n_qubits)
    rho_hat = rho_hat.full()
    rho_hat = rho_hat/np.trace(rho_hat)
    # print(np.trace(rho_hat.full()))
    shadow_estimate = jnp.trace(rho_hat @ obs)
    gt_property = jnp.trace(rho_gt.full() @ obs)
    print(gt_property, shadow_estimate)
    return np.array(gt_property), np.array(shadow_estimate)


def experiment(list_N_shadow, list_n_qubits, reps, obs_fun, key):
    gt_properties = np.zeros((len(list_n_qubits), len(list_N_shadow), reps))
    pred_properties = np.zeros((len(list_n_qubits), len(list_N_shadow), reps))
    keys = jax.random.split(key, gt_properties.shape)
    for i, n_qubits in enumerate(list_n_qubits):
        obs = obs_fun(n_qubits)
        obs_string = ["Z" for _ in range(n_qubits)] # can also change this
        obs = observable_from_string(obs_string)
        for j, N_shadow in enumerate(list_N_shadow):
            for k in range(reps):
                print("n: ", n_qubits, " N: ", N_shadow, " rep: ", k)
                rho_gt = qt.rand_dm([[2]*n_qubits])  # Random density matrix
                gt_properties[i,j,k] , pred_properties[i,j,k] = shadow_estimate(rho_gt, obs, N_shadow, keys[i,j,k])

    return gt_properties, pred_properties

def main():
    Ns = [50, 100, 200, 400, 800, 1600]
    ns = list(range(4,5))
    # Ns = [50, 100]
    # ns = [4]
    seed = 1234
    key = jax.random.PRNGKey(seed)
    reps = 20
    gt_local, pred_local = experiment(Ns, ns, reps, k_local_pauli(key, "Z", 2), key)
    gt_global, pred_global = experiment(Ns, ns, reps, all_one_pauli("Z"), key)

    errors_local = np.sqrt(np.mean(np.abs(gt_local - pred_local)**2, axis=-1))
    errors_global = np.sqrt(np.mean(np.abs(gt_global - pred_global)**2, axis=-1))
    
    for i, n in enumerate(ns):
        y_loc = errors_local[i,:]
        y_glob = errors_global[i,:]
        plt.plot(Ns, y_loc, label="{} qubits, local Z".format(n))
        plt.plot(Ns, y_glob, label="{} qubits, global Z".format(n))

    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("res2.pdf")

if __name__ == "__main__":
    main()