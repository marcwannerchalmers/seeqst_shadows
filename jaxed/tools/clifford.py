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
            lambda s: s._Hadamard(i),
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

    def CNOT(self, c: int | Array, t: int | Array, condition: Array=jnp.array(True)):
        return cond(
            condition,
            lambda s: s._CNOT(c,t),
            lambda s: s,
            self,
        )

    def CZ(self, c: int | Array, t: int | Array, condition: Array=jnp.array(True)):
        return cond(
            condition,
            lambda s: s._CZ(c,t),
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

    def MultiPauliRot(self, theta: Array, paulivector: Array):
        axis_theta = 0 if theta.ndim > 0 else None # shape is static
        x, z, r_vec = vmap(self._single_PauliRot, 
                        in_axes=(axis_theta, 0,1,1),
                        out_axes=(1,1,1))(theta,
                                          paulivector, 
                                        self.tableau[:,:self.n],
                                        self.tableau[:,self.n:])

        tableau = jnp.concatenate([x,z], axis=1)
        r = (self.r + jnp.sum(r_vec, axis=1)) % 2
        return self.replace(tableau=tableau, r=r)

    def PauliRot(self, i: Array, theta: Array, pauli: Array):
        x, z, r = self._single_PauliRot(theta,
                                        pauli,
                                        self.tableau[:,i],
                                        self.tableau[:,i+self.n])

        tableau = self.tableau.at[:,i].set(x)
        tableau = tableau.at[:,i+self.n].set(z)
        r = (self.r + r) % 2
        return self.replace(tableau=tableau, r=r)

    def MultiPauli(self, paulivector: Array):
        _, _, r_vec = vmap(self._single_Pauli, 
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

    def _CNOT(self, control: int | Array, target: int | Array):
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
    def _single_PauliRot(theta: Array, pauli: Array, xi: Array, zi: Array):
        r90 = lambda: Tableau._single_Pauli90(pauli, xi, zi)
        rm90 = lambda: Tableau._single_Paulim90(pauli, xi, zi)
        rpi = lambda: Tableau._single_Pauli(pauli, xi, zi)
        rpihalf = lambda: cond(theta == jnp.pi/2, 
                               r90, 
                               rm90)
        admissible = lambda: cond(jnp.abs(theta) == jnp.pi,
                                  rpi,
                                  rpihalf)
        
        # TODO: Add 0 to the admissible ones
        return admissible()

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

        return xi, zi, x_phase + z_phase


    @staticmethod
    def _single_Pauli90(pauli: Array, xi: Array, zi: Array):
        # The symplectic part is the same for +/- pi/2; only the sign differs.
        xi_res = cond(pauli % 3 == 0,
                      lambda: xi, # Id, RZ
                      lambda: cond(pauli == 2,
                                   lambda: zi, # RY
                                   lambda: (xi + zi) % 2) # RX
                      )

        zi_res = cond(pauli <= 1,
                      lambda: zi, # Id, RX
                      lambda: cond(pauli == 2,
                                   lambda: xi, # RY
                                   lambda: (xi + zi) % 2) # RZ
                      )
        
        r_res = cond(pauli < 2,
                     lambda: cond(pauli == 0,
                                  lambda: jnp.zeros_like(xi), # Id
                                  lambda: zi*(1-xi) # RX
                                  ),
                     lambda: cond(pauli == 2,
                                  lambda: xi*(1-zi), # RY
                                  lambda: xi*zi # RZ
                                  )
                                )
        
        return xi_res, zi_res, r_res

    @staticmethod
    def _single_Paulim90(pauli: Array, xi: Array, zi: Array):
        xi_res = cond(pauli % 3 == 0,
                        lambda: xi, # Id,RY,RZ
                        lambda: cond(pauli==2, lambda: zi, lambda: (xi + zi) % 2) # RX,RZ
                        )
 
        zi_res = cond(pauli <= 1, 
                        lambda: zi, # Id, RX
                        lambda: cond(pauli==2, lambda: xi, lambda: (xi + zi) % 2) # RX,RZ
                        )
        
        r_res = cond(pauli < 2,
                        lambda: cond(pauli == 0,
                                    lambda: jnp.zeros_like(xi), # Id
                                    lambda: xi*zi # RX
                                    ),
                        lambda: cond(pauli == 2,
                                    lambda: zi*(1-xi), # RY
                                    lambda: xi*(1-zi) # RZ
                                    )
                                )
        
        return xi_res, zi_res, r_res


    def _CZ(self, control: int | Array, target: int | Array):
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

    def sample(self, key: Array):
        n = self.n
        keys = jax.random.split(key, n)
        x = self.tableau[:, :n]
        z = self.tableau[:, n:]
        outcome = jnp.zeros((n,), dtype=int)

        def body_fun(a, val):
            x, z, r, outcome = val
            oc, x, z, r = Tableau._single_post_measurement_state(
                keys[a], x, z, r, a, n
            )
            return x, z, r, outcome.at[a].set(oc)

        _, _, _, outcome = fori_loop(0, n, body_fun, (x, z, self.r, outcome))
        return outcome

    @staticmethod
    def _single_basis_sample(key, x: Array, z: Array, r: Array, a: Array, n: int):
        oc, _, _, _ = Tableau._single_post_measurement_state(key, x, z, r, a, n)
        return oc

    @staticmethod
    def _single_post_measurement_state(key: Array, x: Array, z: Array, r: Array, a: Array, n: int):
        condition = (x[n:,a] == 1).any()
        def case1(x, z, r):
            # `argmax` returns the first stabilizer row with x[p, a] == 1.
            p = jnp.argmax(x[n:,a]) + n
            xp, zp, rp = x[p], z[p], r[p]

            def cond_rowsum(i, xh, zh, rh):
                return cond(
                    (i != p) & (xh[a] == 1),
                    lambda: Tableau._rowsum(xh, zh, rh, xp, zp, rp),
                    lambda: (xh, zh, rh),
                )

            x, z, r = vmap(cond_rowsum)(jnp.arange(2*n), x, z, r)

            x, z, r = x.at[p-n].set(x[p]), z.at[p-n].set(z[p]), r.at[p-n].set(r[p])
            x, z = x.at[p].set(0), z.at[p].set(0)
            z = z.at[p,a].set(1)
            r = r.at[p].set(jax.random.randint(key, (), 0, 2))
            return r[p], x, z, r

        def case2(x, z, r):
            init_val = (
                jnp.zeros((n,), dtype=x.dtype),
                jnp.zeros((n,), dtype=z.dtype),
                jnp.array(0, dtype=r.dtype),
            )

            def body_fun(i, scratch):
                return cond(
                    x[i,a] == 1,
                    lambda: Tableau._rowsum(
                        *scratch, x[i+n], z[i+n], r[i+n]
                    ),
                    lambda: scratch,
                )

            _, _, rh = fori_loop(0, n, body_fun, init_val)

            return rh, x,z,r

        oc, x,z,r = cond(condition,
                     case1,
                     case2,
                     x,z,r)

        return oc, x,z,r



    @staticmethod
    def _rowsum(xh: Array, zh: Array, rh: Array,
                xi: Array, zi: Array, ri: Array):

        def g(x1,z1,x2,z2):
            return cond((x1 == 0),
                 lambda: cond(z1==0,
                              lambda: 0, # x1 == 0 and z1 == 0
                              lambda: x2*(1-2*z2)), # x1 == 0 and z1 == 1
                 lambda: cond(z1==0,
                              lambda: z2*(2*x2-1), # x1 == 1 and z1 == 0
                              lambda: z2-x2) # x1 == 1 and z1 == 1
                 )

        g_vals = vmap(g)(xi,zi,xh,zh)
        phase = (2*rh + 2*ri + jnp.sum(g_vals, axis=-1)) % 4
        rh = phase // 2
        return (xh + xi) % 2, (zh + zi) % 2, rh


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

def F(tableau: Tableau, pauli_indices: Array, Gamma: Array, Delta: Array)->Tableau:
    n = Gamma.shape[0]

    # Match utils.F: traverse the lower triangles in descending order.
    def body_iCX(k: int, tableau: Tableau):
        i = n - 1 - k

        def body_j(l: int, tableau: Tableau):
            j = i - 1 - l
            return tableau.CNOT(i, j, Delta[i,j] == 1)

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCX, tableau)

    def body_iCZ(k: int, tableau: Tableau):
        i = n - 1 - k

        def body_j(l: int, tableau: Tableau):
            j = i - 1 - l
            return tableau.CZ(i, j, Gamma[i,j] == 1)

        return fori_loop(0, i, body_j, tableau)

    tableau = fori_loop(0, n, body_iCZ, tableau)
    tableau = tableau.MultiPauli(pauli_indices)
    tableau = tableau.MultiS(Gamma.diagonal())
    return tableau

def canonical_form(tableau: Tableau, Gamma: Array, Delta: Array, 
                   Gammad: Array, Deltad: Array, 
                   h: Array, pauli_indices: Array,
                   S: Array) -> Tableau:
    n = Gamma.shape[0]
    tableau = F(tableau, pauli_indices, Gammad, Deltad)
    tableau = tableau.Permute(S)
    tableau = tableau.MultiHadamard(h)
    tableau = F(tableau, jnp.zeros((n,), dtype=int), Gamma, Delta)

    return tableau

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

def GHZ_type_state_clifford_rev(selective_block: Array,
                            xy: Array,
                            tableau: Tableau):
    
    n = tableau.n
    # Read utils.parallel_entangler_blocks for more explanation
    sorted_indices = jnp.argsort(selective_block, descending=True) 
    sorted_vals = selective_block[sorted_indices] 

    # theta = cond(reversed, lambda: -jnp.pi/2, lambda: jnp.pi/2)
    theta = -jnp.pi/2
    # applies nothing if indices are all 0
    tableau = tableau.PauliRot(jnp.array(sorted_indices[0], dtype=int), 
                                theta, 
                                sorted_vals[0]*(xy+1)) 

    def body_fun(i, tableau: Tableau):
        return tableau.CNOT(sorted_indices[i],
                            sorted_indices[i+1],
                            (sorted_vals[i] == 1) & (sorted_vals[i+1] == 1))

    tableau = fori_loop(0, n-1, body_fun, tableau)

    return tableau

def GHZ_type_state_clifford(selective_block: Array,
                            xy: Array,
                            tableau: Tableau):
    
    n = tableau.n
    # Read utils.parallel_entangler_blocks for more explanation
    sorted_indices = jnp.argsort(selective_block, descending=True) 
    sorted_vals = selective_block[sorted_indices] 
    
    def body_fun(i, tableau: Tableau):
        ind = n-2-i # reversed order 
        return tableau.CNOT(sorted_indices[ind],
                            sorted_indices[ind+1],
                            (sorted_vals[ind] == 1) & (sorted_vals[ind+1] == 1))
    
    tableau = fori_loop(0, n-1, body_fun, tableau)
    
    theta = jnp.pi/2
    tableau = tableau.PauliRot(jnp.array(sorted_indices[0], dtype=int), 
                               theta, 
                               sorted_vals[0]*(xy+1)) # applies nothing if indices are all 0
    
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
