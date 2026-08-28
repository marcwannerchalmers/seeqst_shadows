from __future__ import annotations
import jax.numpy as jnp
from flax import struct
import jax
from jax import Array, jit, vmap
from jax.lax import cond, fori_loop

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from jaxed.tools.binary_solver import solve_binary, stabilizer_phase
from jaxed.tools.observable import PauliObservable
import pennylane as qp
from jaxed.tools.utils import post_meas_state_gates, permutation_to_swaps


# DISCLAIMER: Can be tuned by using bool arrays
# NOTE: one always needs to call tableau = tableau.OP(...)
class Tableau(struct.PyTreeNode):

    tableau: Array 
    r: Array 

    # Initialize as empty array for lazy init
    @classmethod
    def create(cls, n: int=0, N: int=0, N_rho: int=0):
        tableau = jnp.eye(2*n, dtype=int)
        r = jnp.zeros((2*n), dtype=int)
        if N > 0:
            tableau = jnp.stack([tableau for _ in range(N)])
            r = jnp.stack([r for _ in range(N)])
        if N_rho > 0:
            tableau = jnp.stack([tableau for _ in range(N_rho)])
            r = jnp.stack([r for _ in range(N_rho)])

        return cls(tableau=tableau, r=r)

    @property
    def n(self):
        return self.r.shape[-1] // 2

    def H(self, i, condition: Array=jnp.array(True)):
        return cond(
            condition,
            lambda s: s._H(i),
            lambda s: s,
            self,
        )

    def S(self, i, condition: Array=jnp.array(True)):
        return cond(
            condition,
            lambda s: s._S(i),
            lambda s: s,
            self,
        )

    def CNOT(self, i: int, j: int, condition: Array=jnp.array(True)):
        return cond(
            condition,
            lambda s: s._CNOT(i,j),
            lambda s: s,
            self,
        )

    def CZ(self, i: int, j: int, condition: Array=jnp.array(True)):
        return cond(
            condition,
            lambda s: s._CZ(i,j),
            lambda s: s,
            self,
        )

    def MultiS(self, bit_encoding: Array):
        x, z, r_vec = vmap(self._cond_S_chunk, 
                           in_axes=(1,1,0), 
                           out_axes=(1,1,1))(self.tableau[:,:self.n],
                                             self.tableau[:,self.n:],
                                             bit_encoding)

        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1)) % 2
        return self.replace(tableau=tableau, r=r)

    def MultiSdag(self, bit_encoding: Array):
        x, z, r_vec = vmap(self._cond_Sdag_chunk, 
                            in_axes=(1,1,0), 
                            out_axes=(1,1,1))(self.tableau[:,:self.n],
                                                self.tableau[:,self.n:],
                                                bit_encoding)

        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1)) % 2
        return self.replace(tableau=tableau, r=r)

    def MultiHadamard(self, bit_encoding: Array):
        x, z, r_vec = vmap(self._cond_H_chunk, 
                            in_axes=(1,1,0), 
                            out_axes=(1,1,1))(self.tableau[:,:self.n],
                                                self.tableau[:,self.n:],
                                                bit_encoding)
        
        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1)) % 2
        return self.replace(tableau=tableau, r=r)

    def Permute(self, permutation: Array):
        tableau = jnp.concatenate([self.tableau[:,permutation],
                                   self.tableau[:,permutation + self.n]],
                                   axis=1)
        
        return self.replace(tableau=tableau)

    def MultiPauli(self, paulivector: Array):
        r_vec = vmap(self._single_Pauli, 
                     in_axes=(0,1,1),
                     out_axes=1)(paulivector, 
                                self.tableau[:,:self.n],
                                self.tableau[:,self.n:])

        r = (self.r + jnp.sum(r_vec, axis=1)) % 2
        return self.replace(r=r)

    def _Hadamard(self, i: int):
        xi = self.tableau[:,i]
        zi = self.tableau[:,i+self.n]
        r = (self.r + xi*zi) % 2
        tmp = xi
        tableau = self.tableau.at[:,i].set(zi)
        tableau = tableau.at[:,i+self.n].set(tmp)

        return self.replace(tableau=tableau, r=r)

    def _CNOT(self, control: int, target: int):
        xi = self.tableau[:,control]
        zi = self.tableau[:,control+self.n]
        xj = self.tableau[:,target]
        zj = self.tableau[:,target+self.n]

        r = (self.r + xi*zj*((xj+zi+1)%2)) % 2
        xj = (xj + xi) % 2
        zi = (zi + zj) % 2

        tableau = self.tableau.at[:,target].set(xj)
        tableau = tableau.at[:,control+self.n].set(zi)
        return cond(control != target, 
                    lambda: self.replace(tableau=tableau, r=r),
                    lambda: self)
        
    def _S(self, i: int):
        xi = self.tableau[:,i]
        zi = self.tableau[:,i+self.n]

        r = (self.r + xi*zi) % 2
        tableau = self.tableau.at[:,i+self.n].set((zi+xi)%2)

        return self.replace(tableau=tableau, r=r)

    @staticmethod
    def _S_chunk(xi, zi):
        return xi, (zi+xi)%2, (xi*zi) % 2

    @staticmethod
    def _cond_S_chunk(xi, zi, condition):
        return cond(condition, 
                    Tableau._S_chunk,
                    lambda xi, zi: (xi, zi, jnp.zeros_like(xi)),
                    xi, zi)

    @staticmethod
    def _Sdag_chunk(xi, zi):
        return xi, (zi+xi)%2, xi * ((zi + 1) % 2)

    @staticmethod
    def _cond_Sdag_chunk(xi, zi, condition):
        return cond(condition, 
                    Tableau._Sdag_chunk,
                    lambda xi, zi: (xi, zi, jnp.zeros_like(xi)),
                    xi, zi)

    @staticmethod
    def _H_chunk(xi, zi):
        return zi, xi, (xi*zi)

    @staticmethod
    def _cond_H_chunk(xi, zi, condition):
        return cond(condition, 
                    Tableau._H_chunk,
                    lambda xi, zi: (xi, zi, jnp.zeros_like(xi)),
                    xi, zi)

    @staticmethod
    def _single_Pauli(pauli: Array, xi: Array, zi: Array):
        # add x-phase for Y, Z
        x_phase = cond(pauli >= 2, 
                       lambda x: x,
                       lambda x: jnp.zeros_like(x),
                       xi)

        # add z-phase for X,Y
        z_phase = cond((pauli % 3) >= 1, 
                       lambda x: x,
                       lambda x: jnp.zeros_like(x),
                       zi)

        return x_phase + z_phase

    @staticmethod
    def _single_Pauli90(pauli: Array, xi: Array, zi: Array):
        xi_res = cond(pauli != 2,
                      lambda: xi, # Id,RX,RZ
                      lambda: (xi + zi) % 2 # RY
                      )

        zi_res = cond(pauli % 2 == 0, 
                      lambda: zi, # Id, RY
                      lambda: (xi + zi) % 2 # RX,RZ
                      )
        r_res = cond(pauli < 2,
                     lambda: cond(pauli == 0,
                                  lambda: 0, # Id
                                  lambda: zi*(1-xi) # RX
                                  ),
                     lambda: cond(pauli == 2,
                                  lambda: xi*(1-zi), # RY
                                  lambda: xi*zi # RZ
                                  )
                                )
        
        return xi_res, zi_res, r_res


    def _CZ(self, control: int, target: int):
        xi = self.tableau[:,control]
        zi = self.tableau[:,control+self.n]
        xj = self.tableau[:,target]
        zj = self.tableau[:,target+self.n]

        r = (self.r + xi*xj*((zi + zj)%2)) % 2
        tableau = self.tableau.at[:,control+self.n].set((zi+xj)%2)
        tableau = tableau.at[:,target+self.n].set((zj+xi)%2)

        return cond(control != target, 
                    lambda: self.replace(tableau=tableau, r=r),
                    lambda: self)

    # Takes int array
    def expval(self, paulivector: Array):
        p = pauli_to_symp(paulivector)
        M = self.tableau[self.n:].T
        a, contained = solve_binary(M, p)
        return cond(contained,
                    lambda: (-1)**stabilizer_phase(M,self.r[self.n:],a, p),
                    lambda: 0)


# Convention for Pauli vector : I: 0, X: 1, Y: 2, Z: 3
# TODO: Change the convention everywhere in the future
@jit
def symplectic_to_pauli(x: Array,z: Array):
    return cond((x == 0) & (z== 0), 
                lambda x,z: 0,
                lambda x,z: (2*x + z + 1) % 3 + 1,
                x, z) 

def pauli_to_symp(pauli_vector):
    def _single_pauli_to_symp(p: Array):
        return (((p+3)%4)//2+1)%2, p//2

    return jnp.concat(vmap(_single_pauli_to_symp)(pauli_vector))

def _tableau_row_to_paulivector(row_x: Array, row_z: Array): 
    return vmap(symplectic_to_pauli)(row_x,row_z)

def _all_equal(x: Array, y: Array):
    return (x == y).all()

def F_rev(tableau: Tableau, pauli_indices: Array, Gamma: Array, Delta: Array)->Tableau:
    n = Gamma.shape[0]
    tableau = tableau.MultiSdag(Gamma.diagonal())
    tableau = tableau.MultiPauli(pauli_indices)
    # Delta is lower triangular
    def body_iCZ(i: int, tableau: Tableau):
        def body_j(j: int, tableau: Tableau):
            return tableau.CZ(i,j, Gamma[i,j] == 1) # reversed loop

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCZ, tableau)

    # Delta is lower triangular
    def body_iCX(i: int, tableau: Tableau):
        def body_j(j: int, tableau: Tableau):
            return tableau.CNOT(i,j, Delta[i,j] == 1) # reversed loop

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCX, tableau)
    return tableau

def canonical_form_rev(tableau: Tableau, Gamma: Array, Delta: Array, 
                   Gammad: Array, Deltad: Array, 
                   h: Array, pauli_indices: Array,
                   S: Array) -> Tableau:
    n = Gamma.shape[0]
    tableau = F_rev(tableau, jnp.zeros((n,), dtype=int), Gamma, Delta)
    tableau = tableau.MultiHadamard(h)
    tableau = tableau.Permute(jnp.argsort(S))

    tableau = F_rev(tableau, pauli_indices, Gammad, Deltad)

    return tableau


def test_sim():
    n = 5
    N = 10000
    obs = PauliObservable.init_random(jax.random.PRNGKey(12345), n=n,N=N)
    obs = [obs.replace(params=obs.params[i]) for i in range(N)]
    obs[0] = obs[0].replace(params=jnp.ones_like(obs[0].params))
    obs[1] = obs[1].replace(params=2*jnp.ones_like(obs[0].params))
    obs[2] = obs[2].replace(params=3*jnp.ones_like(obs[0].params))
    outcomes = jax.random.randint(jax.random.PRNGKey(12345), (N,n), 0,2, dtype=int)
    condition_vecs = jax.random.randint(jax.random.PRNGKey(0), (N,n), 0,2, dtype=int)
    cnot_pairs = jax.random.randint(jax.random.PRNGKey(12345),(N,n,2), 0, n)
    cnot_conditions = jax.random.randint(jax.random.PRNGKey(54321), (N,n), 0,2, dtype=int)
    S_conditions = jax.random.randint(jax.random.PRNGKey(6789), (N,n), 0,2, dtype=int)
    cz_pairs = jax.random.randint(jax.random.PRNGKey(53637),(N,n,2), 0, n)
    cz_conditions = jax.random.randint(jax.random.PRNGKey(10084), (N,n), 0,2, dtype=int)
    pauli_vecs = jax.random.randint(jax.random.PRNGKey(9382), (N,n), 0,4, dtype=int)
    perm = jax.random.permutation(jax.random.PRNGKey(1434), jnp.stack([jnp.arange(n)]*N), axis=1,independent=True)

    print(perm[0:3])
    # condition_vecs = jnp.zeros_like(condition_vecs)
    @qp.qjit(autograph=True)
    @qp.qnode(qp.device("lightning.qubit", wires=range(n)))
    def circuit_H(obs, outcome, condition_vec, cnot_pair, cnot_cond, S_cond,
                  cz, cz_cond, p, seq):
        post_meas_state_gates(outcome)
        for i in range(n):
            if condition_vec[i] == 1:
                qp.Hadamard(i)
        for i in range(n):
            if cnot_pair[i,0] != cnot_pair[i,1] and cnot_cond[i]==1:
                qp.CNOT(jnp.array([cnot_pair[i,0], cnot_pair[i,1]]))
        for i in range(n):
            if S_cond[i] == 1:
                qp.adjoint(qp.S)(i)

        swap_indices = permutation_to_swaps(seq)
        for i in range(n):
            if swap_indices[i,0] != swap_indices[i,1]:
                qp.SWAP(wires=jnp.stack([swap_indices[i,0], 
                                            swap_indices[i,1]]))
        for i in range(n):
            if cz[i,0] != cz[i,1] and cz_cond[i]==1:
                qp.CZ(jnp.array([cz[i,0], cz[i,1]]))
        for i in range(n):
            if p[i] == 1:
                qp.X(i)
            elif p[i] == 2:
                qp.Y(i)
            elif p[i] == 3:
                qp.Z(i)

        obs.circuit()
        return qp.expval(obs.op())

    def circuit_H_cliff(obs: PauliObservable, outcome, condition_vec,
                        cnot_pair, cnot_cond, S_cond,
                        cz, cz_cond, p, seq):
        tableau = Tableau.create(n)
        tableau = tableau.MultiPauli(outcome)
        tableau = tableau.MultiHadamard(condition_vec)
        def body_j(i: int, tableau: Tableau):
            return tableau.CNOT(cnot_pair[i,0], cnot_pair[i,1], cnot_cond[i]==1) 
        tableau = fori_loop(0, n, body_j, tableau)
        tableau = tableau.MultiSdag(S_cond)
        tableau = tableau.Permute(jnp.argsort(seq))
        def body_k(i: int, tableau: Tableau):
            return tableau.CZ(cz[i,0], cz[i,1], cz_cond[i]==1) 
        tableau = fori_loop(0, n, body_k, tableau)
        tableau = tableau.MultiPauli(p)
        return tableau.expval(obs.params)

    for i in range(N):
        out = circuit_H(obs[i], outcomes[i], condition_vecs[i], cnot_pairs[i],
                        cnot_conditions[i], S_conditions[i], cz_pairs[i],cz_conditions[i],
                        pauli_vecs[i], perm[i])
        out_cliff = circuit_H_cliff(obs[i], outcomes[i], condition_vecs[i], cnot_pairs[i],
                        cnot_conditions[i], S_conditions[i], cz_pairs[i],cz_conditions[i],
                        pauli_vecs[i], perm[i])


        if jnp.abs(out) > 0:
            print(out, out_cliff)
        if jnp.abs(out - out_cliff) > 0.0001:
            print(i, outcomes[i], condition_vecs[i], obs[i].params, cnot_pairs[i],
                        cnot_conditions[i], S_conditions[i], cz_pairs[i], cz_conditions[i],
                        pauli_vecs[i])
            print("Seq", perm[i], perm[i][::-1])


def individual_test():
    n = 5
    tableau = Tableau.create(n)
    paulivec = jnp.array([0,0,3,3,3])
    multi_paulivec = jnp.array([0,0,1,1,1])
    tableau = tableau.MultiPauli(multi_paulivec)
    tableau = tableau.MultiHadamard(jnp.zeros_like(paulivec))
    print(tableau.tableau)
    print(tableau.r)
    print(tableau.expval(paulivec))

def test_new():
    n = 5

    wires = list(range(n))

    vec = jnp.eye(2**n)[:, 0]

    U = (
         qp.S(4)
        @ qp.S(1)
        @ qp.S(0)
        @ qp.CNOT([2,1])
        @ qp.CNOT([4,0])
        @ qp.Hadamard(4)
        @ qp.Hadamard(3)
        @ qp.Hadamard(0)
        @ qp.X(2)
        @ qp.X(3)
    ).matrix(wire_order=wires)

    Y0Z2X3 = (
        qp.Y(0) @ qp.Z(2) @ qp.X(3) @ qp.Identity([1,4])
    ).matrix(wire_order=wires)

    vec = U @ vec

    print(vec.conj().T @ Y0Z2X3 @ vec)

    n = 5
    wires = list(range(n))

    I = jnp.eye(2**n)

    U = I

    # chronological order
    U = qp.X(2).matrix(wire_order=wires) @ U
    U = qp.X(3).matrix(wire_order=wires) @ U

    U = qp.Hadamard(0).matrix(wire_order=wires) @ U
    U = qp.Hadamard(3).matrix(wire_order=wires) @ U
    U = qp.Hadamard(4).matrix(wire_order=wires) @ U

    U = qp.CNOT([4,0]).matrix(wire_order=wires) @ U
    U = qp.CNOT([2,1]).matrix(wire_order=wires) @ U

    U = qp.S(0).matrix(wire_order=wires) @ U
    U = qp.S(1).matrix(wire_order=wires) @ U
    U = qp.S(4).matrix(wire_order=wires) @ U

    vec = U @ jnp.eye(2**n)[:,0]

    Y0Z2X3 = (
        qp.Y(0) @ qp.Z(2) @ qp.X(3) @ qp.Identity([1,4])
    ).matrix(wire_order=wires)

    print(vec.conj().T @ Y0Z2X3 @ vec)

    @qp.qnode(qp.device("lightning.qubit", wires=range(5)))
    def circuit_state(outcome):
        post_meas_state_gates(outcome)

        qp.Hadamard(0)
        qp.Hadamard(3)
        qp.Hadamard(4)

        qp.CNOT([4, 0])
        qp.CNOT([2, 1])

        qp.S(0)
        qp.S(1)
        qp.S(4)

        return qp.expval(qp.Y(0) @ qp.Z(2) @ qp.X(3))

    print(circuit_state(jnp.array([0,0,1,1,0])))

if __name__ == "__main__":
    with jax.disable_jit():
        test_sim()
    """with jax.disable_jit():
        individual_test()"""
    # test_new()