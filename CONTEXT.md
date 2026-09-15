# seeqst_shadows debugging handoff

## Goal

Optimize classical-shadow sampling in `seeqst_shadows` for GPU execution, while preserving the existing PennyLane/Catalyst circuit definitions and scaling eventually to roughly 20 qubits.

The current research code uses PennyLane + Catalyst + JAX. The main issue is that classical shadows require a different measurement setting for essentially every sample, so there is little opportunity to group repeated settings. The desired optimization is therefore **concurrent execution of many independent small/medium statevector circuits**, not reuse of identical measurement settings.

## Repository / runtime context

Repository path used on the cluster:

```text
/nobackup/proj/disk/naiss2026-4-1165/personal/wanner/seeqst_shadows
```

Relevant files:

```text
jaxed/shadow.py
jaxed/experiments.py
experiments_jaxed.py
jaxed/tools/utils.py
jaxed/tools/observable.py
jaxed/tools/state.py
```

Observed software versions during debugging:

```text
jax                     0.7.1
jaxlib                  0.7.1
jax-cuda12-pjrt         0.7.1
jax-cuda12-plugin       0.7.1
pennylane               0.45.1
pennylane_catalyst      0.15.0
pennylane_lightning     0.45.0
Python                  3.13
```

GPU used in the latest run:

```text
NVIDIA GH200 120GB
Driver 580.159.04
CUDA 13.0 reported by nvidia-smi
```

The experiment uses `device="lightning.gpu"`.

## Experiment architecture

`ShadowScalingExperiment.run_batched()` already batches three dimensions:

```text
state_batch_size
bs_obs
bs_N
```

For sample batching:

```text
N_batches = total_N // bs_N
```

Inside the state loop, the state object is constructed once, then `loop_N` processes sample batches of size `bs_N`.

Therefore **`bs_N` should be treated as the desired quantum-execution concurrency width** if asynchronous QNode execution is used. Do not add another 4/8-way chunking layer inside `_sample_circuit`; that would duplicate the batching already controlled by `run_batched()`.

Typical testing call at the moment:

```python
example_experiment_batched(16, 14, 1, None, 128)
```

meaning roughly:

```text
n = 16
k = 14
state batch = 1
observable batch = all
sample batch / concurrency target = 128
```

## Original sampling implementation

The original sampling path was approximately:

```python
@staticmethod
@lru_cache(None)
def _sample_circuit(n: int, U_treedef, state_treedef, device: str):
    U_axes = U_treedef.unflatten(
        [None] * U_treedef.num_leaves
    )
    state_axes = state_treedef.unflatten(
        [None] * state_treedef.num_leaves
    )

    @qp.set_shots(1)
    @qp.qnode(qp.device(device, wires=range(n)))
    def circuit(ind, U, state):
        state()
        U(ind)
        return qp.sample()

    return qjit(autograph=True)(
        catalyst.vmap(
            circuit,
            in_axes=(0, U_axes, state_axes),
        )
    )
```

This is cached with `@lru_cache(None)`, so device/QNode construction is not recreated for every shot.

However, Catalyst documents `catalyst.vmap` as lowering to Catalyst-compatible compiled loops. It should not be assumed to mean true concurrent GPU execution of independent statevectors.

## Why GPU utilization was poor

For classical-shadow sampling, every `(state, measurement setting)` pair can be unique. Under the old implementation, a tiny one-shot QNode was effectively executed repeatedly.

For small `n`, the statevector is tiny. For example:

```text
n = 7 -> 128 amplitudes
```

so GPU kernel-launch/runtime overhead can dominate.

For larger `n`, up to about 20 qubits, dense full-unitary matrix multiplication is not acceptable because it changes scaling from gate-wise statevector simulation (`O(depth * 2^n)`) toward dense matrix costs (`O(4^n)`).

Therefore the preferred direction is to keep Catalyst/Lightning statevector simulation and expose **independent QNode calls concurrently**.

## `jax.vmap(qjit(circuit))` experiment

A test was made with:

```python
compiled_circuit = qjit(circuit, autograph=True)
return jax.vmap(
    compiled_circuit,
    in_axes=(0, U_axes, state_axes),
)
```

Important distinction:

```text
qjit(jax.vmap(...))
```

is not the desired pattern for quantum operations.

Catalyst supports the opposite ordering:

```text
jax.vmap(qjit(circuit))
```

but current Catalyst JAX integration wraps the compiled function through callback machinery and uses sequential vmap behavior. Therefore this route can work functionally but should not be expected to provide true parallel GPU QNode execution.

Also, when compiling QNodes that call dynamic Python control flow in helper functions such as:

```python
if Delta[i, j] == 1:
```

use:

```python
qjit(circuit, autograph=True)
```

because `ind` is dynamic at trace time.

For deeply nested helper functions, `autograph_include` may be needed if AutoGraph does not recursively transform them.

## Current promising direction: explicit async QNodes

Catalyst provides:

```python
qjit(async_qnodes=True)
```

which asks the runtime to execute independent QNode call sites asynchronously when supported by the device/runtime.

The current prototype replaces `catalyst.vmap(circuit, ...)` with **`N` explicit QNode calls**, where `N == bs_N` inside the batched experiment.

Recommended implementation:

```python
@staticmethod
@lru_cache(None)
def _sample_circuit(
    n: int,
    N: int,
    U_treedef,
    state_treedef,
    device: str,
):
    dev = qp.device(
        device,
        wires=range(n),
    )

    @qp.set_shots(1)
    @qp.qnode(dev)
    def circuit(ind, U, state):
        state()
        U(ind)
        return qp.sample()

    @partial(
        qjit,
        autograph=True,
        async_qnodes=True,
    )
    def circuits(indices, U, state):
        return jnp.stack(
            [
                circuit(
                    indices[i],
                    U,
                    state,
                )[0]
                for i in range(N)
            ],
            axis=0,
        )

    return circuits
```

This makes the compiled graph contain explicit independent QNode call sites:

```text
circuit(indices[0], ...)
circuit(indices[1], ...)
...
circuit(indices[N-1], ...)
```

instead of a single mapped QNode in a compiled loop.

### Required `_sample()` shape contract

With the explicit implementation above, each scalar QNode returns approximately:

```text
(1, n)
```

because shots=1.

Inside `_sample_circuit`, `[0]` removes the singleton shot axis:

```text
(n,)
```

and `jnp.stack(..., axis=0)` produces:

```text
(N, n)
```

Therefore `_sample()` must **not** do another `[:, 0]`.

Correct `_sample()`:

```python
def _sample(self, indices, state):
    U_treedef = jax.tree_util.tree_structure(self.U)
    state_treedef = jax.tree_util.tree_structure(state)

    circuit = self._sample_circuit(
        self.n,
        self.N,
        U_treedef,
        state_treedef,
        self.device,
    )

    outcomes = circuit(
        indices,
        self.U,
        state,
    )

    return outcomes
```

Then outer state batching gives:

```text
(N_state_reps, N, n)
```

which is the expected `Shadow.outcomes` shape.

### Important bug already encountered

This was wrong:

```python
return jnp.concatenate(
    [
        circuit(indices[i], U, state)[0]
        for i in range(N)
    ],
    axis=0,
)
```

because each element is `(n,)`, so concatenate produces:

```text
(N * n,)
```

not `(N, n)`.

That caused:

```text
IndexError: Too many indices: 1-dimensional array indexed with 2 regular indices
```

Use `jnp.stack`, not `jnp.concatenate`, if `[0]` is applied to each QNode result.

## Current uploaded-code inconsistency to fix first

The latest uploaded `shadow.py` still showed both of these stale lines:

```python
outcomes = circuit(indices, self.U, state)[:,0]
```

and:

```python
return jnp.concatenate(
    [
        circuit(...)[0]
        for i in range(N)
    ],
    axis=0,
)
```

Before benchmarking async performance, change them to the `stack` + no-`[:,0]` version shown above.

## Outer state batching

`sample_statevector()` currently does:

```python
create_samples = lambda shadow, indices, state: shadow._sample(indices, state)

outcomes = jax.vmap(
    create_samples,
    in_axes=(None, 0, 0),
)(
    self,
    self.indices,
    states,
)
```

For now, leave this alone.

If:

```text
state_batch_size = 1
bs_N = 128
```

then there are 128 explicit async QNode calls.

If later `state_batch_size > 1`, there is another JAX vmap around `_sample`; only optimize/flatten that dimension after the inner async strategy is validated.

## GPU memory observation

Latest `nvidia-smi` snapshot during the async experiment:

```text
GPU: NVIDIA GH200 120GB
Memory: ~78.7 GiB used by python
GPU util: ~7%
Power: ~152 W / 900 W
```

Do **not** interpret the ~79 GiB as live quantum statevectors.

JAX normally preallocates roughly 75% of visible GPU memory. On a ~97.9 GiB visible GPU, that alone is about 73.4 GiB.

For a 16-qubit statevector:

```text
2^16 amplitudes * 16 bytes/complex128 ~= 1 MiB
```

so 128 statevectors are only roughly 128 MiB before workspaces.

For debugging real memory use, run with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments_jaxed.py
```

This is especially important because JAX and Lightning-GPU both need memory from the same GPU.

## GPU-utilization interpretation

A single `nvidia-smi` snapshot is insufficient because it may catch Catalyst/XLA compilation, which is mostly CPU-side.

However, if sustained GPU utilization remains around 7% during actual quantum execution at `bs_N=128`, then the async QNode approach is probably **not** producing useful GPU concurrency.

The main possibilities are:

```text
1. QNodes overlap at the host/runtime level but Lightning GPU work serializes.
2. Lightning/Catalyst runtime does not expose separate CUDA streams for these QNodes.
3. Calls overlap but each quantum kernel is still too small to saturate the GH200.
4. Measurement was taken during compilation rather than execution.
```

## Benchmark plan

Do not redesign again before measuring.

Benchmark:

```text
bs_N = 1
bs_N = 2
bs_N = 4
bs_N = 8
bs_N = 16
bs_N = 32
bs_N = 64
bs_N = 128
```

Use the same total number of samples for each test.

Measure **steady-state execution after compilation**, not total wall time including first compilation.

A useful concurrency signature would look like:

```text
bs_N     samples/sec
1        x
2        ~2x
4        ~4x
8        increasing
16       increasing
32       saturation begins
64       small gain
128      little/no further gain
```

If runtime instead scales roughly linearly with `bs_N`, execution is effectively serialized.

## Nsight Systems test

If available on the cluster, this is the decisive diagnostic:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
nsys profile \
    -t cuda,nvtx,osrt \
    -o async_bs128 \
    python experiments_jaxed.py
```

Inspect the CUDA timeline.

Desired behavior:

```text
stream 1: [quantum kernels]...
stream 2:   [quantum kernels]...
stream 3: [quantum kernels]...
stream 4:    [quantum kernels]...
```

with overlapping time ranges.

Bad behavior:

```text
single stream:
[circuit 0][circuit 1][circuit 2]...[circuit 127]
```

If everything is serialized onto one CUDA stream, `async_qnodes=True` is not solving the core performance problem.

## Why not dense-unitary batching

A previously considered prototype was to convert each existing PennyLane measurement circuit to a dense `2^n x 2^n` unitary and batch matrix-vector multiplication in JAX.

Do not pursue this for the long-term 20-qubit target.

It is acceptable only as a small-`n` benchmark because dense matrices scale as `4^n` in memory and computation.

The desired long-term scaling is ordinary gate-wise statevector simulation:

```text
O(depth * 2^n)
```

with concurrency over independent circuits.

## cuStateVec note

The capability needed by the project exists at the cuStateVec level: NVIDIA exposes batched statevector APIs that can apply small gate matrices to many independent statevectors and perform batched measurement.

The limitation is that current PennyLane/Catalyst `vmap` does not automatically lower the classical-shadow workload into those batched cuStateVec APIs.

If Catalyst async QNodes fail to provide real concurrency, a later fallback could be a thin custom cuStateVec batched execution layer. However, that would require more backend work and should only be considered after profiling the async-QNode route.

## Other relevant correctness/performance notes

### `Shadow.init` wasted allocation

Base `Shadow.init` currently allocates:

```python
snapshots=Tableau.create(n, N, N_state_reps)
```

but `create_snapshots()` immediately replaces it.

For `PauliShadow`, `create_snapshots()` is a no-op, so the base tableau allocation is entirely unnecessary.

Since the field type allows `None`, initialize with:

```python
snapshots=None
```

unless some subclass actually requires the preallocated tableau.

### `outcomes=jnp.empty(...)`

`Shadow.init` also creates an empty outcomes array that sampling immediately replaces. This is less straightforward to remove because `n`, `N`, and `N_state_reps` are currently properties derived from `outcomes.shape`.

### Pauli reconstruction

`PauliShadow._inverse_circuit` is already pure JAX and avoids PennyLane QNode execution for reconstruction. This is good and should remain.

### `estimate_weak_properties` condition bug

Base implementation currently uses something like:

```python
condition = obs.params.shape[0] == self.N_state_reps
```

This confuses `bs_obs == state_batch_size` with “different observables per state.” It is currently harmless when those dimensions differ but is a latent correctness bug.

### Ground truth

At one point the batched `run_batched()` implementation returned an unchanged zero `gt_vec`, meaning GT was not actually computed in that refactored path. Verify the current local version before using results scientifically.

## Current immediate next action

1. Fix the async sampling output shape:

```python
jnp.stack([... circuit(...)[0] ...], axis=0)
```

and remove the old `[:,0]` in `_sample()`.

2. Run with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

3. Benchmark `bs_N = 1, 8, 32, 128` first.

4. Time after compilation.

5. If GPU utilization remains low, run `nsys profile` and inspect whether CUDA kernels from independent QNodes actually overlap.

6. Only if async QNodes do not create real overlap, investigate a lower-level cuStateVec batched backend.

## Design principle to preserve

The main goal is to avoid rewriting the scientific circuit definitions.

Keep existing abstractions such as:

```text
State
PartialCircuit
SEEQSTShadow.U_fun
SEEQSTShadow.Udag_fun
PauliShadow.U_fun
CliffordShadow.U_fun
PauliObservable
Estimator
```

and change only the execution backend / sampling path wherever possible.
