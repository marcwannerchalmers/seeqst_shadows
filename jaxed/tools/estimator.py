from typing import Any
import jax
from jax import numpy as jnp
from jax import Array
from flax import struct

@struct.dataclass
class Estimator:

    def __call__(self, x: Array):
        return jnp.mean(x, axis=-1)
    
class MedianOfMeans(Estimator):
    k: int = struct.field(pytree_node=False)

    def __call__(self, x: Array):
        buckets = jnp.array_split(x, self.k, axis=-1)
        buckets = jnp.stack([super().__call__(bucket) for bucket in buckets], axis=-1)
        return jnp.median(buckets, axis=-1)

