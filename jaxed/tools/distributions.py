from jax import numpy as jnp
import jax 
from jax import Array, random
from typing import List
from jax.lax import cond

def uniform(key: Array, 
            N: int, n: int, 
            sample_idx_range: List[int], 
            full_setting: bool=False)->Array:

    def full_fn(key: Array, indices: Array):
        N2 = N//2
        init_indices = random.randint(key, (N2,n), *sample_idx_range)
        indices = indices.at[:N2,:n].set(init_indices)
        indices = indices.at[N2:2*N2,:n].set(init_indices)
        indices = indices.at[:N2,n].set(jnp.zeros((init_indices.shape[0],), dtype=indices.dtype))
        indices = indices.at[N2:2*N2,n].set(jnp.ones((init_indices.shape[0],), dtype=indices.dtype))
        return indices

    def rand_fn(key: Array, indices: Array):
        key1, key2 = random.split(key)
        init_indices = random.randint(key1, (N,n), *sample_idx_range)
        setting_indices = random.randint(key2, (N,), 0, 2)
        indices = indices.at[:,:n].set(init_indices)
        indices = indices.at[:,n].set(setting_indices)

        return indices

    indices = jnp.zeros((N,n+1), dtype=int)

    return cond(full_setting, full_fn, rand_fn, key, indices)

# TODO: binomial distribution