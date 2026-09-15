import numpy as np
import jax
from jax import numpy as jnp
from jax import random, jit


def _apply_single_qubit_gates(statevectors, gates, wire: int, n: int):
    """Apply one (possibly different) gate per batched statevector."""
    left_dim = 1 << wire
    right_dim = 1 << (n - wire - 1)
    shaped = statevectors.reshape(
        (*statevectors.shape[:-1], left_dim, 2, right_dim)
    )
    transformed = jnp.einsum("...ab,...lbr->...lar", gates, shaped)
    return transformed.reshape(statevectors.shape)


def _apply_static_cnot(statevectors, control: int, target: int, condition, n: int):
    basis = jnp.arange(1 << n, dtype=jnp.int32)
    control_bit = (basis >> (n - control - 1)) & 1
    permutation = basis ^ (control_bit << (n - target - 1))
    transformed = jnp.take(statevectors, permutation, axis=-1)
    return jnp.where(condition[..., None], transformed, statevectors)


def _apply_dynamic_cnot(statevectors, controls, targets, condition, n: int):
    basis = jnp.arange(1 << n, dtype=jnp.int32)
    control_masks = jnp.left_shift(
        jnp.ones_like(controls, dtype=jnp.int32),
        (n - controls - 1).astype(jnp.int32),
    )
    target_masks = jnp.left_shift(
        jnp.ones_like(targets, dtype=jnp.int32),
        (n - targets - 1).astype(jnp.int32),
    )
    control_is_one = (
        basis[None, None, :] & control_masks[..., None]
    ) != 0
    permutation = basis[None, None, :] ^ jnp.where(
        control_is_one,
        target_masks[..., None],
        0,
    )
    transformed = jnp.take_along_axis(statevectors, permutation, axis=-1)
    return jnp.where(condition[..., None], transformed, statevectors)


def _apply_static_cz(statevectors, wire_a: int, wire_b: int, condition, n: int):
    basis = jnp.arange(1 << n, dtype=jnp.int32)
    both_one = (
        ((basis >> (n - wire_a - 1)) & 1)
        * ((basis >> (n - wire_b - 1)) & 1)
    )
    phase = (1 - 2 * both_one).astype(statevectors.dtype)
    transformed = statevectors * phase
    return jnp.where(condition[..., None], transformed, statevectors)


def _apply_dynamic_swap(statevectors, wire_a, wire_b, condition, n: int):
    basis = jnp.arange(1 << n, dtype=jnp.int32)
    mask_a = jnp.left_shift(
        jnp.ones_like(wire_a, dtype=jnp.int32),
        (n - wire_a - 1).astype(jnp.int32),
    )
    mask_b = jnp.left_shift(
        jnp.ones_like(wire_b, dtype=jnp.int32),
        (n - wire_b - 1).astype(jnp.int32),
    )
    different = (
        ((basis[None, None, :] & mask_a[..., None]) != 0)
        != ((basis[None, None, :] & mask_b[..., None]) != 0)
    )
    permutation = basis[None, None, :] ^ jnp.where(
        different,
        mask_a[..., None] | mask_b[..., None],
        0,
    )
    transformed = jnp.take_along_axis(statevectors, permutation, axis=-1)
    return jnp.where(condition[..., None], transformed, statevectors)


def _single_qubit_gate_tables(dtype):
    inv_sqrt_two = jnp.asarray(1 / np.sqrt(2), dtype=dtype)
    identity = jnp.eye(2, dtype=dtype)
    hadamard = inv_sqrt_two * jnp.asarray([[1, 1], [1, -1]], dtype=dtype)
    hadamard_sdag = inv_sqrt_two * jnp.asarray(
        [[1, -1j], [1, 1j]],
        dtype=dtype,
    )
    paulis = jnp.asarray(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=dtype,
    )
    phase = jnp.asarray([[1, 0], [0, 1j]], dtype=dtype)
    return identity, hadamard, hadamard_sdag, paulis, phase


def _sample_computational_basis(key, statevectors, n: int, output_dtype):
    probabilities = jnp.real(statevectors * jnp.conj(statevectors))
    cumulative = jnp.cumsum(probabilities, axis=-1)
    uniforms = random.uniform(
        key,
        cumulative.shape[:-1],
        dtype=probabilities.dtype,
    )
    basis_samples = jnp.sum(
        cumulative < uniforms[..., None],
        axis=-1,
    )
    basis_samples = jnp.minimum(basis_samples, (1 << n) - 1)
    shifts = jnp.arange(n - 1, -1, -1, dtype=basis_samples.dtype)
    return ((basis_samples[..., None] >> shifts) & 1).astype(output_dtype)


@jit
def _sample_pauli_statevectors(key, indices, input_statevectors):
    n = indices.shape[-1]
    N = indices.shape[-2]
    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (input_statevectors.shape[0], N, input_statevectors.shape[-1]),
    )
    identity, hadamard, hadamard_sdag, _, _ = _single_qubit_gate_tables(
        statevectors.dtype
    )
    measurement_gates = jnp.stack(
        [hadamard, hadamard_sdag, identity],
        axis=0,
    )

    for wire in range(n):
        gates = measurement_gates[indices[..., wire]]
        statevectors = _apply_single_qubit_gates(
            statevectors,
            gates,
            wire,
            n,
        )

    return _sample_computational_basis(
        key,
        statevectors,
        n,
        indices.dtype,
    )


@jit
def _sample_seeqst_statevectors(key, indices, input_statevectors):
    n = indices.shape[-1] - 1
    N = indices.shape[-2]
    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (input_statevectors.shape[0], N, input_statevectors.shape[-1]),
    )
    blocks = indices[..., :n]
    xy_settings = indices[..., n]
    sorted_wires = jnp.argsort(-blocks, axis=-1, stable=True)
    sorted_values = jnp.take_along_axis(blocks, sorted_wires, axis=-1)

    for position in range(n - 2, -1, -1):
        active_pair = (
            (sorted_values[..., position] == 1)
            & (sorted_values[..., position + 1] == 1)
        )
        statevectors = _apply_dynamic_cnot(
            statevectors,
            sorted_wires[..., position],
            sorted_wires[..., position + 1],
            active_pair,
            n,
        )

    real_dtype = jnp.real(statevectors).dtype
    cosine = jnp.asarray(np.cos(np.pi / 4), dtype=real_dtype)
    sine = jnp.asarray(np.sin(np.pi / 4), dtype=real_dtype)
    rx = jnp.asarray(
        [[cosine, -1j * sine], [-1j * sine, cosine]],
        dtype=statevectors.dtype,
    )
    ry = jnp.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=statevectors.dtype,
    )
    rotations = jnp.where(
        (xy_settings == 0)[..., None, None],
        rx,
        ry,
    )
    active_rotation = sorted_values[..., 0] == 1
    rotation_wires = sorted_wires[..., 0]

    for wire in range(n):
        transformed = _apply_single_qubit_gates(
            statevectors,
            rotations,
            wire,
            n,
        )
        statevectors = jnp.where(
            (active_rotation & (rotation_wires == wire))[..., None],
            transformed,
            statevectors,
        )

    return _sample_computational_basis(
        key,
        statevectors,
        n,
        indices.dtype,
    )


@jit
def _sample_clifford_statevectors(key, indices, input_statevectors):
    n = indices.shape[-2]
    N = indices.shape[1]
    statevectors = jnp.broadcast_to(
        input_statevectors[:, None, :],
        (input_statevectors.shape[0], N, input_statevectors.shape[-1]),
    )
    gamma = indices[..., :, :n]
    delta = indices[..., :, n : 2 * n]
    gamma_dag = indices[..., :, 2 * n : 3 * n]
    delta_dag = indices[..., :, 3 * n : 4 * n]
    hadamard_flags = indices[..., :, 4 * n]
    pauli_indices = indices[..., :, 4 * n + 1]
    swaps = indices[..., :, 4 * n + 2 : 4 * n + 4]

    identity, hadamard, _, paulis, phase = _single_qubit_gate_tables(
        statevectors.dtype
    )

    def apply_f(statevectors, pauli_values, gamma_values, delta_values):
        for row in range(n - 1, -1, -1):
            for column in range(row - 1, -1, -1):
                statevectors = _apply_static_cnot(
                    statevectors,
                    row,
                    column,
                    delta_values[..., row, column] == 1,
                    n,
                )

        for row in range(n - 1, -1, -1):
            for column in range(row - 1, -1, -1):
                statevectors = _apply_static_cz(
                    statevectors,
                    row,
                    column,
                    gamma_values[..., row, column] == 1,
                    n,
                )

        for wire in range(n):
            statevectors = _apply_single_qubit_gates(
                statevectors,
                paulis[pauli_values[..., wire]],
                wire,
                n,
            )
            phase_gates = jnp.where(
                (gamma_values[..., wire, wire] == 1)[..., None, None],
                phase,
                identity,
            )
            statevectors = _apply_single_qubit_gates(
                statevectors,
                phase_gates,
                wire,
                n,
            )
        return statevectors

    statevectors = apply_f(
        statevectors,
        pauli_indices,
        gamma_dag,
        delta_dag,
    )

    for swap_index in range(n - 1, -1, -1):
        wire_a = swaps[..., swap_index, 0]
        wire_b = swaps[..., swap_index, 1]
        statevectors = _apply_dynamic_swap(
            statevectors,
            wire_a,
            wire_b,
            wire_a != wire_b,
            n,
        )

    for wire in range(n):
        gates = jnp.where(
            (hadamard_flags[..., wire] == 1)[..., None, None],
            hadamard,
            identity,
        )
        statevectors = _apply_single_qubit_gates(
            statevectors,
            gates,
            wire,
            n,
        )

    statevectors = apply_f(
        statevectors,
        jnp.zeros_like(pauli_indices),
        gamma,
        delta,
    )
    return _sample_computational_basis(
        key,
        statevectors,
        n,
        indices.dtype,
    )

