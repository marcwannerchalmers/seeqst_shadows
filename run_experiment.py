"""Hydra entry point for declarative shadow-scaling experiments."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from hydra import main as hydra_main
from jax import random
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.optimize import minimize_scalar

from experiments_jaxed import GHZ_Klocal
from jaxed.experiments import Experiments, ShadowScalingExperiment
from jaxed.shadow import CliffordShadow, PauliShadow, SEEQSTShadow
from jaxed.tools.distributions import binomial, fixed_weight, uniform
from jaxed.tools.majorana import (
    FermionicGaussianShadow,
    MajoranaState,
    majorana_observables_list,
)
from jaxed.tools.noise import depolarizing_noise
from jaxed.tools.observable import PauliObservable
from jaxed.tools.state import GHZType, HRState


SHADOW_CLASSES = {
    "SEEQSTShadow": SEEQSTShadow,
    "PauliShadow": PauliShadow,
    "CliffordShadow": CliffordShadow,
    "FermionicGaussianShadow": FermionicGaussianShadow,
}

STATE_CLASSES = {
    "HRState": HRState,
    "GHZType": GHZType,
    "MajoranaState": MajoranaState,
}

PAULI_INDICES = {
    "I": 0,
    "X": 1,
    "Y": 2,
    "Z": 3,
}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass
class ExperimentPlan:
    """Parsed experiments plus their plotting configuration."""

    experiments: list[ShadowScalingExperiment]
    plot: dict[str, Any]
    context: dict[str, Any]


class _FormatContext(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _as_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_config(cfg: DictConfig | Mapping[str, Any] | str | Path) -> DictConfig:
    if isinstance(cfg, (str, Path)):
        loaded = OmegaConf.load(cfg)
    elif isinstance(cfg, DictConfig):
        loaded = cfg
    else:
        loaded = OmegaConf.create(cfg)
    OmegaConf.resolve(loaded)
    return loaded


def _evaluate_expression(expression: str, context: Mapping[str, Any]) -> Any:
    """Evaluate arithmetic expressions containing only configured variables."""

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in context:
            return context[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Unsupported expression: {expression!r}")

    return evaluate(ast.parse(expression, mode="eval"))


def _resolve_value(value: Any, context: Mapping[str, Any]) -> Any:
    value = _as_plain(value)
    if not isinstance(value, str):
        return value
    if value in context:
        return context[value]
    try:
        return _evaluate_expression(value, context)
    except (KeyError, SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return value.format_map(_FormatContext(context))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _integer_sequence(spec: Any, context: Mapping[str, Any]) -> list[int]:
    spec = _as_plain(spec)
    if isinstance(spec, Sequence) and not isinstance(spec, str):
        return [int(_resolve_value(value, context)) for value in spec]
    if not isinstance(spec, Mapping):
        return [int(_resolve_value(spec, context))]
    if "values" in spec:
        return _integer_sequence(spec["values"], context)
    if "range" in spec:
        range_spec = spec["range"]
        start = int(_resolve_value(range_spec.get("start", 0), context))
        stop = int(_resolve_value(range_spec["stop"], context))
        step = int(_resolve_value(range_spec.get("step", 1), context))
        return list(range(start, stop, step))
    if "powers" in spec:
        power_spec = spec["powers"]
        base = int(_resolve_value(power_spec.get("base", 2), context))
        start = int(_resolve_value(power_spec["start"], context))
        stop = int(_resolve_value(power_spec["stop"], context))
        step = int(_resolve_value(power_spec.get("step", 1), context))
        return [base**power for power in range(start, stop, step)]
    raise ValueError(f"Expected values, range, or powers, got: {spec}")


def _pauli_indices(values: Sequence[str | int]) -> list[int]:
    indices = []
    for value in values:
        if isinstance(value, str):
            try:
                value = PAULI_INDICES[value.upper()]
            except KeyError as exc:
                raise ValueError(f"Unknown Pauli label {value!r}") from exc
        indices.append(int(value))
    return indices


def _observable_lists(
    spec: Mapping[str, Any],
    ns: list[int],
    key,
    context: Mapping[str, Any],
) -> list[PauliObservable]:
    factory = spec.get("factory", spec.get("init", "random_pauli"))
    if factory in {"random", "random_pauli"}:
        count = int(_resolve_value(spec["count"], context))
        sample_indices = _pauli_indices(spec.get("sample_indices", ["I", "X", "Y", "Z"]))
        padding_indices = _pauli_indices(spec.get("padding_indices", ["I"]))
        keys = random.split(key, len(ns))
        observables = []
        for obs_key, n in zip(keys, ns):
            local_context = {**context, "n": n}
            k_local = int(_resolve_value(spec.get("k_local", 0), local_context))
            observables.append(
                PauliObservable.init_random(
                    obs_key,
                    sample_indices=sample_indices,
                    n=n,
                    N=count,
                    k_local=k_local,
                    padding_indices=padding_indices,
                )
            )
        return observables

    if factory in {"majorana", "majorana_rdm"}:
        order = int(_resolve_value(spec.get("order", context.get("k")), context))
        return majorana_observables_list(ns, order)

    if factory in {"file", "npy"} or str(factory).endswith(".npy"):
        path = spec.get("path", factory)
        params = np.load(str(path))
        if params.ndim == 2 and len(ns) == 1:
            params = params[None, ...]
        if params.ndim != 3 or params.shape[0] != len(ns):
            raise ValueError(
                "Observable files must have shape (len(ns), N_obs, n), "
                "or (N_obs, n) for one n"
            )
        return [PauliObservable.init(jnp.asarray(values, dtype=jnp.int32)) for values in params]

    raise ValueError(f"Unknown observable factory {factory!r}")


def _state_class(spec: Mapping[str, Any], context: Mapping[str, Any]):
    class_name = spec.get("class", "HRState")
    if class_name == "GHZ_Klocal":
        locality = int(_resolve_value(spec["locality"], context))
        return GHZ_Klocal(locality)
    try:
        return STATE_CLASSES[class_name]
    except KeyError as exc:
        raise ValueError(f"Unknown state class {class_name!r}") from exc


def _distribution(
    spec: str | Mapping[str, Any],
    n: int,
    observables: PauliObservable,
    context: Mapping[str, Any],
):
    if isinstance(spec, str):
        name = spec
        spec = {}
    else:
        name = spec.get("name", "uniform")
    if name == "uniform":
        return uniform
    if name == "fixed_weight":
        if spec.get("weights", "rdm_optim") != "rdm_optim":
            raise ValueError("fixed_weight currently supports weights: rdm_optim")
        return partial(
            fixed_weight,
            probabilities=_rdm_fixed_weight_probabilities(n, observables),
        )
    if name != "binomial":
        raise ValueError(f"Unknown distribution {name!r}")

    q_spec = spec.get("q")
    if q_spec == "rdm_optim":
        q = _rdm_optimized_q(n, observables)
    elif q_spec == "mean_xy_support":
        xy_support = (observables.params == 1) | (observables.params == 2)
        q = float(jnp.mean(jnp.sum(xy_support, axis=1)) / n)
        q = min(max(q, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))
    else:
        q = float(_resolve_value(q_spec, {**context, "n": n}))
    if not 0.0 < q < 1.0:
        raise ValueError(f"Binomial q must lie strictly between zero and one, got {q}")
    return partial(binomial, q=q)


def _rdm_optimized_q(n: int, observables: PauliObservable) -> float:
    """Minimize the exact average inverse-channel eigenvalue for the targets."""
    params = np.asarray(observables.params)
    n_xy = np.count_nonzero((params == 1) | (params == 2), axis=1)
    n_z = np.count_nonzero(params == 3, axis=1)
    non_z = n_xy > 0

    def risk(q: float) -> float:
        non_z_eigenvalues = 0.5 * q**n_xy[non_z] * (1.0 - q)**(
            n - n_xy[non_z]
        )
        z_eigenvalues = 0.5 * (
            1.0 + (1.0 - 2.0 * q) ** n_z[~non_z]
        )
        return float(
            np.sum(1.0 / non_z_eigenvalues) + np.sum(1.0 / z_eigenvalues)
        )

    optimum = minimize_scalar(
        risk,
        bounds=(1.0e-6, 1.0 - 1.0e-6),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not optimum.success:
        raise ValueError(f"Could not optimize the RDM binomial q: {optimum.message}")
    return float(optimum.x)


def _rdm_fixed_weight_probabilities(
    n: int, observables: PauliObservable
) -> tuple[float, ...]:
    """Closed-form average-optimal fixed-weight mixture for the non-Z sector."""
    params = np.asarray(observables.params)
    supports = np.count_nonzero((params == 1) | (params == 2), axis=1)
    probabilities = np.zeros(n + 1, dtype=np.float64)
    for support in np.unique(supports[supports > 0]):
        count = np.count_nonzero(supports == support)
        probabilities[support] = np.sqrt(count * math.comb(n, int(support)))
    if probabilities.sum() == 0:
        raise ValueError("fixed_weight requires at least one non-Z observable")
    probabilities /= probabilities.sum()
    return tuple(float(value) for value in probabilities)


def _fixed_weight_eigenvalues(
    n: int, probabilities: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """SEEQST channel eigenvalues for an exchangeable fixed-weight mixture."""
    non_z = np.zeros(n + 1, dtype=np.float64)
    z_type = np.zeros(n + 1, dtype=np.float64)
    for support in range(1, n + 1):
        non_z[support] = (
            probabilities[support] / (2.0 * math.comb(n, support))
        )
    for z_weight in range(n + 1):
        visibility = 0.0
        for weight, probability in enumerate(probabilities):
            even_probability = sum(
                math.comb(z_weight, overlap)
                * math.comb(n - z_weight, weight - overlap)
                for overlap in range(0, z_weight + 1, 2)
                if 0 <= weight - overlap <= n - z_weight
            ) / math.comb(n, weight)
            visibility += probability * even_probability
        z_type[z_weight] = visibility
    return tuple(non_z), tuple(z_type)


def _shadow_arguments(
    spec: Mapping[str, Any] | None,
    ns: list[int],
    observables: list[PauliObservable],
    context: Mapping[str, Any],
):
    raw = dict(spec or {})
    arguments_by_n = {}
    for n, obs in zip(ns, observables):
        local_context = {**context, "n": n}
        parsed = {}
        for name, value in raw.items():
            if name == "distribution":
                parsed[name] = _distribution(value, n, obs, local_context)
                if (
                    isinstance(parsed[name], partial)
                    and parsed[name].func is fixed_weight
                ):
                    probabilities = parsed[name].keywords["probabilities"]
                    parsed["fixed_weight_eigenvalues"] = (
                        _fixed_weight_eigenvalues(n, probabilities)
                    )
            elif name in {"noise", "noise_fun"}:
                noise_spec = _as_plain(value)
                if noise_spec is None or (
                    isinstance(noise_spec, str)
                    and noise_spec.lower() == "none"
                ):
                    continue
                if not isinstance(noise_spec, Mapping):
                    raise ValueError("noise must be null or a mapping")
                if noise_spec.get("name") != "depolarizing":
                    raise ValueError(f"Unknown noise model {noise_spec.get('name')!r}")
                p = float(_resolve_value(noise_spec["p"], local_context))
                parsed["noise_fun"] = partial(depolarizing_noise, p=p)
            else:
                parsed[name] = _resolve_value(value, local_context)
        arguments_by_n[n] = parsed

    def arguments_for_n(n: int) -> dict[str, Any]:
        return arguments_by_n[n]

    return arguments_for_n


def _expanded_experiments(
    specs: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    expanded = []
    for raw_spec in specs:
        spec = dict(raw_spec)
        sweep = spec.pop("for_each", None)
        if sweep is None:
            expanded.append((spec, dict(context)))
            continue
        variable = sweep["variable"]
        for value in _integer_sequence(sweep, context):
            expanded.append((spec, {**context, variable: value}))
    return expanded


def parse_experiment_config(
    cfg: DictConfig | Mapping[str, Any] | str | Path,
) -> ExperimentPlan:
    """Parse a YAML/OmegaConf configuration without executing the experiments."""

    config = _load_config(cfg)
    plain = OmegaConf.to_container(config, resolve=True)
    context = {
        key: value
        for key, value in plain.items()
        if isinstance(value, (int, float, str, bool))
    }

    ns = _integer_sequence(plain["qubits"], context)
    Ns = _integer_sequence(plain["samples"], context)
    if not ns or not Ns:
        raise ValueError("qubits and samples must both be non-empty")
    context.update(
        {
            "nmin": min(ns),
            "nmax": max(ns),
            "Nmin": min(Ns),
            "Nmax": max(Ns),
            "reps": int(plain["repetitions"]),
        }
    )
    if len(ns) == 1:
        context["n"] = ns[0]

    key = random.PRNGKey(int(plain.get("seed", 12345)))
    key, shared_obs_key = random.split(key)
    shared_observable_spec = dict(plain.get("observables", {}))
    shared_observables = None
    if shared_observable_spec:
        shared_observables = _observable_lists(
            shared_observable_spec,
            ns,
            shared_obs_key,
            context,
        )

    batch = dict(plain.get("batch", {}))
    default_state_spec = dict(plain.get("state", {"class": "HRState"}))
    experiments = []
    expanded = _expanded_experiments(plain["experiments"], context)
    for experiment_spec, experiment_context in expanded:
        observable_override = dict(experiment_spec.get("observables", {}))
        if observable_override:
            if not shared_observable_spec:
                observable_spec = observable_override
            else:
                observable_spec = _deep_merge(
                    shared_observable_spec, observable_override
                )
            key, obs_key = random.split(key)
            observables = _observable_lists(
                observable_spec,
                ns,
                obs_key,
                experiment_context,
            )
        elif shared_observables is not None:
            observable_spec = shared_observable_spec
            observables = shared_observables
        else:
            raise ValueError("Each experiment needs an observable specification")

        N_obs = int(observables[0].params.shape[0])
        local_context = {**experiment_context, "N_obs": N_obs}
        state_spec = _deep_merge(
            default_state_spec,
            dict(experiment_spec.get("state", {})),
        )
        state_cls = _state_class(state_spec, local_context)

        shadow_name = experiment_spec["shadow_class"]
        try:
            shadow_cls = SHADOW_CLASSES[shadow_name]
        except KeyError as exc:
            raise ValueError(f"Unknown shadow class {shadow_name!r}") from exc

        key, experiment_key = random.split(key)
        shadow_args = _shadow_arguments(
            experiment_spec.get("shadow_args"),
            ns,
            observables,
            local_context,
        )
        first_distribution = shadow_args(ns[0]).get("distribution")
        if (
            isinstance(first_distribution, partial)
            and first_distribution.func is binomial
        ):
            local_context["q"] = first_distribution.keywords["q"]
        k_local_value = experiment_spec.get(
            "k_local",
            observable_spec.get("order", observable_spec.get("k_local")),
        )
        k_local = (
            None
            if k_local_value is None
            else int(_resolve_value(k_local_value, {**local_context, "n": ns[0]}))
        )
        path_prefix = _resolve_value(
            experiment_spec.get("path_prefix"), local_context
        )
        experiment_name = _resolve_value(
            experiment_spec.get("experiment_name", experiment_spec.get("id")),
            local_context,
        )
        experiments.append(
            ShadowScalingExperiment(
                shadow_cls,
                state_cls,
                shadow_args=shadow_args,
                state_name=state_spec.get("name"),
                N_list=Ns,
                n_list=ns,
                obs_lists=observables,
                N_state_reps=int(plain["repetitions"]),
                verbose=bool(experiment_spec.get("verbose", plain.get("verbose", False))),
                key=experiment_key,
                path_save=path_prefix,
                state_batch_size=batch.get("states", 1),
                batch_size_obs=batch.get("observables"),
                batch_size_N=batch.get("samples"),
                k_local=k_local,
                experiment_name=experiment_name,
            )
        )

    context["N_obs"] = int(experiments[0].observables_list[0].params.shape[0])
    plot = dict(plain.get("plot", {}))
    files = plot.get("files")
    if files:
        entries = files.items() if isinstance(files, Mapping) else (
            next(iter(entry.items())) for entry in files
        )
        experiments_by_name = {
            experiment.experiment_name: experiment for experiment in experiments
        }
        selected = []
        expected_shape = (
            2,
            len(ns),
            int(plain["repetitions"]),
            context["N_obs"],
            len(Ns),
        )
        for name, filename in entries:
            if name not in experiments_by_name:
                raise ValueError(f"No configured experiment named {name!r}")
            path = Path(_resolve_value(filename, context))
            if path.suffix != ".npy" or not path.is_file():
                raise ValueError(f"Plot data file does not exist: {path}")
            if np.load(path, mmap_mode="r").shape != expected_shape:
                raise ValueError(
                    f"Plot data {path} does not have expected shape {expected_shape}"
                )
            experiment = experiments_by_name[name]
            experiment.path_save = str(path.with_suffix(""))
            selected.append(experiment)
        experiments = selected

    for bound in ("N_min", "N_max", "n_min", "n_max"):
        if plot.get(bound) is not None:
            plot[bound] = _resolve_value(plot[bound], context)
    if plot.get("epsilon") is not None:
        plot["epsilon"] = _resolve_value(plot["epsilon"], context)
    if plot.get("path") is not None:
        plot["path"] = _resolve_value(plot["path"], context)
    return ExperimentPlan(experiments=experiments, plot=plot, context=context)


def run_from_config(
    cfg: DictConfig | Mapping[str, Any] | str | Path,
) -> Experiments:
    """Parse, run, and plot a configured collection of experiments."""

    plan = parse_experiment_config(cfg)
    for experiment in plan.experiments:
        if experiment.path_save is not None:
            Path(experiment.path_save).parent.mkdir(parents=True, exist_ok=True)
    experiments = Experiments(plan.experiments, build_dataframe=False)

    plot = plan.plot
    if plot.get("enabled", True):
        plot_method = plot.get("function", "plot_avg_var")
        if plot_method not in {
            "plot_avg_var",
            "plot_avg_var_zero_segments",
            "plot_sample_complexity",
        }:
            raise ValueError(f"Unknown plotting function {plot_method!r}")
        path = plot.get("path")
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        style = dict(plot.get("style", {}))
        kwargs = {
            "path_save": path,
            "xlog": bool(plot.get("xlog", False)),
            "ylog": bool(plot.get("ylog", False)),
            "metric": plot.get("metric", "mae"),
            "aggregation_batch_size": int(
                plot.get("aggregation_batch_size", 16)
            ),
            "progress": bool(plot.get("progress", True)),
            "N_min": plot.get("N_min"),
            "N_max": plot.get("N_max"),
            "n_min": plot.get("n_min"),
            "n_max": plot.get("n_max"),
            **style,
        }
        if plot_method == "plot_sample_complexity":
            if plot.get("epsilon") is None:
                raise ValueError("plot_sample_complexity requires plot.epsilon")
            kwargs["epsilon"] = plot["epsilon"]
        else:
            kwargs["x"] = plot.get("x", "N")
        getattr(experiments, plot_method)(**kwargs)
    return experiments


@hydra_main(
    version_base="1.3",
    config_path="experiment_cfgs",
    config_name="xy_local_paulis",
)
def main(cfg: DictConfig) -> None:
    run_from_config(cfg)


if __name__ == "__main__":
    main()
