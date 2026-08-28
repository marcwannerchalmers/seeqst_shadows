import jax
from jax import numpy as jnp
from jax.lax import fori_loop, cond
from jax import Array
from typing import Tuple
import numpy as np
import galois

def solve_binary(M: jax.Array, p: jax.Array):
    m, n = M.shape
    # We solve A @ a = p with A = M.T.
    # Augmented matrix: [A | p], shape (m, n+1).
    A = jnp.concatenate(
        [M, p[:, None]],
        axis=1,
    )
    def body(k, Aw):
        # Find a pivot row >= k.
        A, w = Aw
        pivot_col = A[:,k]*w
        pivot = jnp.argmax(pivot_col)
        # Take care of the case that all subsequent rows are already triu
        pivot = cond(pivot_col[pivot] == 1,
                     lambda: pivot,
                     lambda: k)

        # Swap row k and pivot.
        row_k = A[k]
        row_p = A[pivot]
        w = w.at[k].set(0)

        A = A.at[k].set(row_p)
        A = A.at[pivot].set(row_k)
        # Eliminate this variable from all rows below.
        factor = A[:,k]*w
        A = (A + jnp.outer(factor, A[k])) % 2

        return A, w

    A, _ = fori_loop(0, n, body, (A,jnp.ones_like(p)))
    # assert jnp.all(jnp.triu(A[:,:-1]) == A[:,:-1])
    # Since A has n independent columns, rows n:m
    # should have zero in the coefficient part.
    inconsistent = jnp.any(
        jnp.all(A[n:, :-1] == 0, axis=1) &
        (A[n:, -1] == 1)
    )
    def back_body(k, a):
        i = n - 1 - k
        rhs = A[i, -1]

        contribution = jnp.sum(
            A[i,:-1] * a
        ) % 2

        a = a.at[i].set(
            (rhs + contribution) % 2
        )

        return a

    a = fori_loop(
        0,
        n,
        back_body,
        jnp.zeros(n, dtype=A.dtype),
    )

    return a, ~inconsistent

def stabilizer_phase(M, r, a, p):
    m, n = M.shape

    x = M[:n]
    z = M[n:]

    # Signs carried by the generators themselves
    phase = jnp.sum(a * r)

    # z_i dot x_j
    pair_int = z.T @ x

    # Only the parity matters for the (-1) pairwise factor
    pair_mod2 = pair_int % 2

    commutation_phase = jnp.sum(
        jnp.triu(
            jnp.outer(a, a) * pair_mod2,
            k=1,
        )
    )

    # IMPORTANT: do NOT use pair_mod2 here.
    # We need the integer number of Y factors.
    generator_y = jnp.sum(
        jnp.diag(
            jnp.outer(a, a) * pair_int
        )
    )

    target_y = jnp.dot(p[:n], p[n:])

    y_components = (generator_y - target_y) // 2

    return (
        phase
        + commutation_phase
        + y_components
    ) % 2

def test_solver():
    n = 5
    N = 100
    random_mats = np.random.randint(0,2,(N,n,n))
    rand_vecs = np.random.randint(0,2,(N,n))
    for i in range(N):
        M, p = random_mats[i], rand_vecs[i]
        a, success = solve_binary(jnp.array(M), jnp.array(p))
        GF2 = galois.GF(2)
        A = GF2(M)
        b = GF2(p)
        try:
            x = np.linalg.solve(A, b)
            print(a,x)
        except np.linalg.LinAlgError as e:
            continue
            # print(e)

def test_expval():
    n = 5
    N = 1000
    random_mats = np.random.randint(0,2,(N,2*n,n))
    rand_vecs = np.random.randint(0,2,(N,2*n))
    r = np.random.randint(0,2,(N,n))
    for i in range(N):
        M, p = random_mats[i], rand_vecs[i]
        a, success = solve_binary(jnp.array(M), jnp.array(p))
        r_p = stabilizer_phase(M, r[i], a)
        if success:
            print(a, success, r_p)

if __name__ == "__main__":
    # test_solver()
    test_expval()