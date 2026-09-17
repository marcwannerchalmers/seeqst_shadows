from __future__ import annotations
import pennylane as qp
import jax
from jax import Array
from jax import numpy as jnp

# TODO: Change to dataclass such that each noise class implements
# a function for each simulator
def depolarizing_noise(p: float, key: Array, n: int):
    probs = jnp.array([
        1 - p,
        p / 3,
        p / 3,
        p / 3,
    ])

    errors = jax.random.choice(
        key,
        4,
        shape=(n,),
        p=probs,
    )

    for wire in range(n):
        error = errors[wire]
        angle = jnp.asarray(jnp.pi, dtype=jnp.float32)
        qp.RX(angle * (error == 1), wires=wire)
        qp.RY(angle * (error == 2), wires=wire)
        qp.RZ(angle * (error == 3), wires=wire)
