import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.lax import cond
from flax import struct


@struct.dataclass
class MeanState:
    total: Array
    count: Array
    values: Array


@struct.dataclass
class MedianOfMeansState:
    total: Array
    count: Array
    prefix_values: Array


@struct.dataclass
class Estimator:

    def __call__(self, x: Array, N: Array = jnp.array(0, dtype=jnp.int32)):
        len_x = x.shape[-1]
        N = cond((N <= 0) | (len_x < N), lambda: len_x, lambda: N)
        mask = jnp.arange(len_x, dtype=jnp.int32) < N
        return jnp.sum(x * mask, axis=-1, dtype=jnp.float32) / jnp.asarray(
            N, dtype=jnp.float32
        )

    def validate_Ns(self, Ns: Array) -> None:
        if np.any(np.asarray(Ns) <= 0):
            raise ValueError("Requested sample counts must be positive")

    def init_state(self, value_shape: tuple[int, ...], Ns: Array) -> MeanState:
        return MeanState(
            total=jnp.zeros(value_shape, dtype=jnp.float32),
            count=jnp.array(0, dtype=jnp.int32),
            values=jnp.zeros(value_shape + (Ns.shape[0],), dtype=jnp.float32),
        )

    def update(self, state: MeanState, x_batch: Array, Ns: Array) -> MeanState:
        batch_prefix = jnp.cumsum(x_batch, axis=-1, dtype=jnp.float32)
        offsets = jnp.clip(Ns - state.count - 1, 0, x_batch.shape[-1] - 1)
        prefix_sums = state.total[..., None] + jnp.take(
            batch_prefix, offsets, axis=-1
        )
        reached = (Ns > state.count) & (
            Ns <= state.count + x_batch.shape[-1]
        )
        shape = (1,) * state.total.ndim + (Ns.shape[0],)
        values = jnp.where(
            reached.reshape(shape),
            prefix_sums / Ns.astype(jnp.float32).reshape(shape),
            state.values,
        )
        return MeanState(
            total=state.total + jnp.sum(x_batch, axis=-1, dtype=jnp.float32),
            count=state.count + x_batch.shape[-1],
            values=values,
        )

    def finalize(self, state: MeanState, Ns: Array) -> Array:
        del Ns
        return state.values


@struct.dataclass
class MedianOfMeans(Estimator):
    k: int = struct.field(pytree_node=False)

    def validate_Ns(self, Ns: Array) -> None:
        super().validate_Ns(Ns)
        if np.any(np.asarray(Ns) < self.k):
            raise ValueError("MedianOfMeans requires every requested N >= k")

    def _boundaries(self, Ns: Array) -> Array:
        Ns = jnp.asarray(Ns, dtype=jnp.int32)
        bucket = jnp.arange(self.k + 1, dtype=jnp.int32)
        quotient, remainder = Ns[:, None] // self.k, Ns[:, None] % self.k
        return bucket * quotient + jnp.minimum(bucket, remainder)

    def __call__(self, x: Array, N: Array = jnp.array(0, dtype=jnp.int32)):
        len_x = x.shape[-1]
        N = cond((N <= 0) | (len_x < N), lambda: len_x, lambda: N)
        boundaries = self._boundaries(jnp.asarray([N]))[0]
        prefix = jnp.concatenate(
            [jnp.zeros(x.shape[:-1] + (1,), x.dtype), jnp.cumsum(x, axis=-1)],
            axis=-1,
        )
        bucket_sums = jnp.diff(jnp.take(prefix, boundaries, axis=-1), axis=-1)
        bucket_sizes = jnp.diff(boundaries).astype(jnp.float32)
        return jnp.median(bucket_sums / bucket_sizes, axis=-1)

    def init_state(
        self, value_shape: tuple[int, ...], Ns: Array
    ) -> MedianOfMeansState:
        return MedianOfMeansState(
            total=jnp.zeros(value_shape, dtype=jnp.float32),
            count=jnp.array(0, dtype=jnp.int32),
            prefix_values=jnp.zeros(
                value_shape + (Ns.shape[0], self.k + 1), dtype=jnp.float32
            ),
        )

    def update(
        self, state: MedianOfMeansState, x_batch: Array, Ns: Array
    ) -> MedianOfMeansState:
        boundaries = self._boundaries(Ns)
        batch_prefix = jnp.cumsum(x_batch, axis=-1, dtype=jnp.float32)
        offsets = jnp.clip(
            boundaries - state.count - 1, 0, x_batch.shape[-1] - 1
        )
        prefix_sums = state.total[..., None, None] + jnp.take(
            batch_prefix, offsets.reshape(-1), axis=-1
        ).reshape(state.prefix_values.shape)
        reached = (boundaries > state.count) & (
            boundaries <= state.count + x_batch.shape[-1]
        )
        shape = (1,) * state.total.ndim + reached.shape
        return MedianOfMeansState(
            total=state.total + jnp.sum(x_batch, axis=-1, dtype=jnp.float32),
            count=state.count + x_batch.shape[-1],
            prefix_values=jnp.where(
                reached.reshape(shape), prefix_sums, state.prefix_values
            ),
        )

    def finalize(self, state: MedianOfMeansState, Ns: Array) -> Array:
        boundaries = self._boundaries(Ns)
        bucket_sums = jnp.diff(state.prefix_values, axis=-1)
        bucket_sizes = jnp.diff(boundaries, axis=-1).astype(jnp.float32)
        shape = (1,) * state.total.ndim + bucket_sizes.shape
        return jnp.median(bucket_sums / bucket_sizes.reshape(shape), axis=-1)
