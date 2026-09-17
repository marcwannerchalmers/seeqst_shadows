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
        init_indices = random.randint(key, (N2,n), *sample_idx_range, dtype=jnp.int32)
        indices = indices.at[:N2,:n].set(init_indices)
        indices = indices.at[N2:2*N2,:n].set(init_indices)
        indices = indices.at[:N2,n].set(jnp.zeros((init_indices.shape[0],), dtype=indices.dtype))
        indices = indices.at[N2:2*N2,n].set(jnp.ones((init_indices.shape[0],), dtype=indices.dtype))
        return indices

    def rand_fn(key: Array, indices: Array):
        key1, key2 = random.split(key)
        init_indices = random.randint(key1, (N,n), *sample_idx_range, dtype=jnp.int32)
        setting_indices = random.randint(key2, (N,), 0, 2, dtype=jnp.int32)
        indices = indices.at[:,:n].set(init_indices)
        indices = indices.at[:,n].set(setting_indices)

        return indices

    indices = jnp.zeros((N,n+1), dtype=jnp.int32)

    return cond(full_setting, full_fn, rand_fn, key, indices)

def binomial(key: Array,
            N: int, n: int,
            sample_idx_range: List[int],
            q)->Array:
    """Sample Bernoulli block masks and one X/Y setting per shot."""
    q = jnp.asarray(q, dtype=jnp.float32)
    key1, key2 = jax.random.split(key)
    block_inds = jax.random.bernoulli(key1, q, (N,n))
    xy = jax.random.bernoulli(key2, shape=(N,))

    return jnp.concatenate([block_inds, xy[:,None]], axis=1).astype(jnp.int32)


def fixed_weight(key: Array,
                 N: int, n: int,
                 sample_idx_range: List[int],
                 probabilities)->Array:
    """Sample a block weight, then a uniformly random block of that weight."""
    probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    key1, key2, key3 = jax.random.split(key, 3)
    weights = jax.random.categorical(
        key1, jnp.log(probabilities), shape=(N,)
    ).astype(jnp.int32)
    priorities = jax.random.uniform(key2, (N, n), dtype=jnp.float32)
    ranks = jnp.argsort(jnp.argsort(priorities, axis=1), axis=1)
    block_inds = ranks < weights[:, None]
    xy = jax.random.bernoulli(key3, shape=(N,))

    return jnp.concatenate([block_inds, xy[:, None]], axis=1).astype(jnp.int32)
