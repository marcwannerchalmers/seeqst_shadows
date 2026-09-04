from typing import Any
import jax
from jax import numpy as jnp
from jax import Array
from flax import struct
from jax.lax import cond

@struct.dataclass
class Estimator:

    def __call__(self, x: Array, N: Array=jnp.array(0)):
        len_x = jnp.ones_like(x, dtype=int).sum()
        N = cond((N <= 0) | (len_x < N),
                 lambda: len_x,
                 lambda: N)
        mask = jnp.where(jnp.arange(x.shape[-1]) < N, 1, 0)
        return jnp.sum(x*mask, axis=-1)/N
    
class MedianOfMeans(Estimator):
    k: int = struct.field(pytree_node=False)

    def __call__(self, x: Array, N: Array=jnp.array(0)):
        len_x = jnp.ones_like(x).sum()
        N = jnp.floor(len_x/self.k)
        buckets = jnp.array_split(x, self.k, axis=-1)
        buckets = jnp.stack([super().__call__(bucket, N) for bucket in buckets], axis=-1)
        return jnp.median(buckets, axis=-1)

def test():
    x = jnp.arange(20)
    Ns = jnp.array([0, 10, 20, 30])
    est = Estimator()
    @jax.jit
    def f(x, N):
        return est(x, N)

    print(jax.vmap(f, in_axes=(None, 0))(x, Ns))

if __name__ == "__main__":
    test()