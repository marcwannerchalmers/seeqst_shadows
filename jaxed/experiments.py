from typing import List, Type
import numpy as np
import jax
from jax import numpy as jnp
from jax import random, vmap, jit, Array
from jax import tree
from jax.lax import fori_loop, dynamic_slice_in_dim, dynamic_update_slice_in_dim
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import partial, lru_cache
import time
from jax_tqdm import loop_tqdm
from tqdm.auto import tqdm

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxed.shadow import PauliShadow, CliffordShadow, SEEQSTShadow, Shadow
from jaxed.tools.estimator import Estimator
from jaxed.tools.observable import PauliObservable, Observable
from jaxed.tools.state import HRState, State
from jaxed.plotting import (
    _normalise_metric,
    plot_avg_var as plot_avg_var_figure,
    plot_avg_var_zero_segments as plot_avg_var_zero_segments_figure,
    plot_sample_complexity as plot_sample_complexity_figure,
)
from GPshadow import GPShadow

jax.config.update(
    "jax_compilation_cache_dir",
    "/nobackup/proj/disk/naiss2026-4-1165/personal/wanner/jax_cache",
)
jax.config.update(
    "jax_persistent_cache_min_compile_time_secs",
    0,
)

# TODO: Save the outcomes and load them if the path exists, printing what it ended up doing.
class Experiments:
    """Run, collect and plot a group of shadow-scaling experiments."""

    def __init__(
        self,
        scaling_experiments: List["ShadowScalingExperiment"],
        name_dict=None,
        build_dataframe=True,
    ) -> None:
        self.experiments = scaling_experiments
        for experiment in scaling_experiments:
            experiment.run()
        self._data = self.get_dataframe() if build_dataframe else None

    @property
    def data(self):
        if self._data is None:
            self._data = self.get_dataframe()
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def get_dataframe(self):
        data = {
            "Shadow model": [],
            "Observable": [],
            "Ground truth": [],
            "Prediction": [],
            "Error": [],
            "N": [],
            "n": [],
            "State": [],
            "experiment_name": [],
        }
        for experiment in self.experiments:
            for j, (observables, n) in enumerate(
                zip(experiment.observables_list, experiment.ns)
            ):
                for state_index in range(experiment.N_state_reps):
                    for observable_index in range(observables.params.shape[0]):
                        for sample_index, N in enumerate(experiment.Ns):
                            model_name = experiment.shadow_cls.__name__
                            if experiment.shadow_cls is GPShadow:
                                model_name = experiment.shadows[0].cfg["kernel_type"]

                            ground_truth = experiment.gt[
                                j, state_index, observable_index, sample_index
                            ]
                            prediction = experiment.preds[
                                j, state_index, observable_index, sample_index
                            ]
                            state_name = (
                                str(experiment.state_cls)
                                if experiment.state_name is None
                                else experiment.state_name
                            )

                            data["Shadow model"].append(model_name)
                            data["Observable"].append(
                                observables.obs_string(
                                    observables.params[observable_index]
                                )
                            )
                            data["Ground truth"].append(ground_truth)
                            data["Prediction"].append(prediction)
                            data["Error"].append(abs(ground_truth - prediction))
                            data["N"].append(N)
                            data["n"].append(n)
                            data["State"].append(
                                f"{state_name}_{state_index}"
                            )
                            data["experiment_name"].append(
                                experiment.experiment_name
                            )

        return pd.DataFrame(
            {key: np.asarray(values) for key, values in data.items()}
        )

    def plot(self, index_list=None, path_save=None, **lineplot_args):
        data = self.data
        if index_list:
            data = data[index_list]
        sns.set_style("whitegrid")
        result = sns.lineplot(data, **lineplot_args)
        result.set(xscale="log")
        plt.show()
        if path_save is not None:
            plt.savefig(path_save)
        return result

    def plot_avg_var(
        self,
        path_save=None,
        xlog=False,
        ylog=False,
        metric="mae",
        x="N",
        aggregation_batch_size=16,
        progress=True,
        N_min=None,
        N_max=None,
        n_min=None,
        n_max=None,
        **style,
    ):
        data = self._bounded_plot_data(
            self.get_plot_dataframe(metric, aggregation_batch_size, progress),
            N_min=N_min,
            N_max=N_max,
            n_min=n_min,
            n_max=n_max,
        )
        return plot_avg_var_figure(
            data,
            path_save=path_save,
            x=x,
            xlog=xlog,
            ylog=ylog,
            metric=metric,
            prepared=True,
            **style,
        )

    def plot_avg_var_zero_segments(
        self,
        path_save=None,
        xlog=False,
        ylog=False,
        metric="mae",
        x="N",
        aggregation_batch_size=16,
        progress=True,
        N_min=None,
        N_max=None,
        n_min=None,
        n_max=None,
        **style,
    ):
        """Plot all-zero prediction runs as dashed, unshaded segments."""

        data = self._bounded_plot_data(
            self.get_plot_dataframe(metric, aggregation_batch_size, progress),
            N_min=N_min,
            N_max=N_max,
            n_min=n_min,
            n_max=n_max,
        )
        return plot_avg_var_zero_segments_figure(
            data,
            path_save=path_save,
            x=x,
            xlog=xlog,
            ylog=ylog,
            metric=metric,
            prepared=True,
            **style,
        )

    def plot_sample_complexity(
        self,
        epsilon,
        path_save=None,
        xlog=False,
        ylog=True,
        metric="rmse",
        aggregation_batch_size=16,
        progress=True,
        N_min=None,
        N_max=None,
        n_min=None,
        n_max=None,
        **style,
    ):
        data = self._bounded_plot_data(
            self.get_plot_dataframe(metric, aggregation_batch_size, progress),
            N_min=N_min,
            N_max=N_max,
            n_min=n_min,
            n_max=n_max,
        )
        return plot_sample_complexity_figure(
            data,
            epsilon=float(epsilon),
            path_save=path_save,
            xlog=xlog,
            ylog=ylog,
            metric=metric,
            **style,
        )

    def get_plot_dataframe(self, metric="mae", N_batch_size=16, progress=True):
        """Aggregate plot statistics directly from result tensors in N batches."""

        metric_name = _normalise_metric(metric)
        if N_batch_size <= 0:
            raise ValueError("aggregation_batch_size must be positive")
        total = sum(
            len(experiment.ns)
            * ((len(experiment.Ns) + N_batch_size - 1) // N_batch_size)
            for experiment in self.experiments
        )
        rows = []
        with tqdm(
            total=total,
            desc="Aggregating plot data",
            unit="batch",
            disable=not progress,
        ) as bar:
            for experiment in self.experiments:
                model_name = experiment.shadow_cls.__name__
                Ns = np.asarray(experiment.Ns)
                for n_index, n in enumerate(experiment.ns):
                    for start in range(0, len(Ns), N_batch_size):
                        stop = min(start + N_batch_size, len(Ns))
                        predictions = np.asarray(
                            jax.device_get(
                                experiment.preds[n_index, :, :, start:stop]
                            ),
                            dtype=np.float64,
                        )
                        ground_truth = np.asarray(
                            jax.device_get(
                                experiment.gt[n_index, :, :, start:stop]
                            ),
                            dtype=np.float64,
                        )
                        errors = np.abs(ground_truth - predictions)
                        if metric_name == "mae":
                            repetition_metric = errors.mean(axis=1)
                        elif metric_name == "rmse":
                            repetition_metric = np.sqrt(
                                np.mean(errors**2, axis=1)
                            )
                        else:
                            repetition_metric = errors.max(axis=1)

                        means = repetition_metric.mean(axis=0)
                        stds = (
                            repetition_metric.std(axis=0, ddof=1)
                            if repetition_metric.shape[0] > 1
                            else np.full(stop - start, np.nan)
                        )
                        all_zero = np.equal(predictions, 0).all(axis=(0, 1))
                        for index, N in enumerate(Ns[start:stop]):
                            rows.append(
                                {
                                    "Shadow model": model_name,
                                    "N": int(N),
                                    "n": int(n),
                                    "experiment_name": experiment.experiment_name,
                                    "mean": means[index],
                                    "std": stds[index],
                                    "count": repetition_metric.shape[0],
                                    "repetition_metric": repetition_metric[:, index],
                                    "all_predictions_zero": bool(all_zero[index]),
                                    "ci": stds[index],
                                    "ci_low": means[index] - stds[index],
                                    "ci_high": means[index] + stds[index],
                                }
                            )
                        bar.update()
        result = pd.DataFrame(rows)
        result.attrs["metric"] = metric_name
        return result

    @staticmethod
    def _bounded_plot_data(data, N_min=None, N_max=None, n_min=None, n_max=None):
        for column, lower, upper in (
            ("N", N_min, N_max),
            ("n", n_min, n_max),
        ):
            if lower is not None:
                data = data[data[column] >= lower]
            if upper is not None:
                data = data[data[column] <= upper]
        if data.empty:
            raise ValueError("No plot data remains within the configured bounds")
        return data

    def _observable_uncertainty_data(self):
        """Average within observables over states; vary across observables."""

        return self.data.rename(
            columns={"State": "Observable", "Observable": "State"}
        )


class ShadowScalingExperiment:
    def __init__(self, shadow_cls: Type[Shadow], state_cls: Type[State], shadow_args=lambda n: {}, state_params=None, state_name=None,
                  N_list=[], n_list=[], obs_lists:List[PauliObservable]=[],
                  gate_indices=lambda n: jnp.array([0,2], dtype=jnp.int32), estimator=lambda n, N: Estimator(),
                  N_state_reps: int=1, key=random.PRNGKey(12345), 
                  verbose=False, state_batch_size=3, path_save=None, k_local=None,
                  batch_size_obs=None,
                  batch_size_N=None,
                  experiment_name=None) -> None:
        self.shadow_cls = shadow_cls
        self.state_cls = state_cls
        self.shadow_args = shadow_args
        self.state_params = state_params
        self.Ns = jnp.array(N_list, dtype=jnp.int32)
        self.N = int(jnp.max(self.Ns))
        self.ns = n_list
        self.N_state_reps = N_state_reps
        self.observables_list = []
        if len(obs_lists) == len(self.ns):
            self.observables_list = obs_lists

        self.k_local = k_local

        self.preds = jnp.zeros((len(self.Ns), len(self.ns), self.observables_list[0].params.shape[0]), dtype=jnp.float32)
        self.gt = jnp.zeros_like(self.preds)
        self.gate_indices = gate_indices
        self.estimator = estimator
        self.state_name = state_name
        self.times = {"Create Shadow": [], "Predict": [], "Compute GT": [], "Total step": []}
        self.verbose = verbose
        self.state_batch_size = state_batch_size
        self.use_batched = batch_size_N is not None or batch_size_obs is not None
        self.bs_N = batch_size_N if batch_size_N is not None else self.N
        self.bs_obs = (batch_size_obs if batch_size_obs is not None
                       else self.observables_list[0].params.shape[0])
        if path_save is None:
            self.path_save = None
        else:
            batch_suffix = ""
            if self.use_batched:
                batch_suffix = (f"_bsState{self.state_batch_size}"
                                f"_bsObs{self.bs_obs}_bsN{self.bs_N}")
            self.path_save = (
                f"{path_save}_nmax{max(self.ns)}"
                f"_Nobs{self.observables_list[0].params.shape[0]}"
                f"_reps{self.N_state_reps}_N{max(self.Ns)}"
                f"_k{self.k_local}{batch_suffix}"
            )
        if self.use_batched:
            self.states = []
            self.shadows = []
        else:
            self.init(key)
        self.key = random.split(key)[1]
        self.experiment_name = experiment_name

    def init(self, key):
        keys1, keys2 = random.split(key, (2,len(self.ns)))
        if self.state_params is None:
            self.states = [self.state_cls.init_random(keys1[i], self.N_state_reps, n)
                                for i, n in enumerate(self.ns)]
        N = self.N

        self.shadows = [self.shadow_cls.init(keys2[i], n, N, self.N_state_reps,
                                              self.gate_indices(n), self.estimator(n,N),
                                              **self.shadow_args(n))
                                for i, n in enumerate(self.ns)]

    
    def run(self):
        save_state = False
        experiments_loaded = False
        if self.path_save is not None:
            print(self.path_save)
            if os.path.isfile(self.path_save+".npy"):
                self.load_results(self.path_save+".npy")
                experiments_loaded = True
            else:
                if self.verbose:
                    print("Experiments will be saved as {}".format(self.path_save))
                save_state = True

        if not experiments_loaded:
            if self.use_batched:
                start = time.perf_counter()
                res = self.run_batched(self.observables_list)
                preds, gt = zip(*res)
                self.preds = jnp.stack(preds, axis=0)
                self.gt = jnp.stack(gt, axis=0)
                jax.block_until_ready((self.preds, self.gt))
                end = time.perf_counter()
                if self.verbose:
                    print("Estimating {} observables from {} states with {} samples took {} s".format(self.observables_list[0].params.shape[0], self.N_state_reps, self.N, end-start))
            else:
                leave_fun = lambda x: isinstance(x, (self.shadow_cls, self.state_cls, Observable))
                # sampling
                # sample_fun = lambda shadow, states: shadow.sample(states)
                """replace_fun = lambda shadow, outcomes: shadow.replace(outcomes=outcomes)
                outcomes = tree.map(sample_fun, self.shadows, self.states) 
                self.shadows = tree.map(replace_fun, self.shadows, outcomes)"""
                # self.shadows = tree.map(sample_fun, self.shadows, self.states,
                                        # is_leaf=leave_fun)
                self.key, key2 = random.split(self.key)
                keys_shad = list(random.split(key2, len(self.ns)))

                @jit
                def sample_shadows(shadows, states):
                    sample_fun = lambda shadow, states, key: shadow.sample(key, states)
                    return tree.map(sample_fun, shadows, states, keys_shad,
                                        is_leaf=leave_fun)

                if self.verbose:
                    print("Sampling {} from {}-{} qubits".format(str(type(self.states[0])).split(".")[-1].split("\'")[0], 
                                                                        min(self.ns), 
                                                                        max(self.ns)))
                start = time.perf_counter()
                self.shadows = sample_shadows(self.shadows, self.states)
                jax.block_until_ready(self.shadows)
                end = time.perf_counter()
                if self.verbose:
                    print("Generating {} states x {} samples = {} total samples took {} s".format(len(self.ns)*self.N_state_reps,
                                                    max(self.Ns),
                                                    len(self.ns)*self.N_state_reps*max(self.Ns),
                                                    end-start))
                
                # prediction
                obs_list = self.observables_list
                # Adapt observable TODO: Put that into shadow class
                #-----------
                # Case 2: Same observables for each state 
                # --> shape_0 of params does not equal N_states for ALL observables
                """case2_fun = lambda obs: obs.params.shape[0] == self.N_state_reps
                case2 = not jnp.array(tree.map(case2_fun, obs_list)).all()
                obs_axes = (None, 0) if case2 else (0,1)"""
                #-----------
                start = time.perf_counter()
                if self.shadow_cls is GPShadow:
                    self.gt, self.preds = self.estimate_gp(obs_list)
                else: 
                    self.preds, self.gt = self.estimate_full(obs_list) 
                jax.block_until_ready((self.preds, self.gt))
                end = time.perf_counter()

                print("Creating {} snapshots and processing them for {} observables each took {} s".format(len(self.ns)*self.N_state_reps*max(self.Ns),
                                                    self.observables_list[0].params.shape[0]*len(self.ns)*self.N_state_reps,
                                                    end-start))

        if save_state:
            self.save_results(self.path_save)

    # TODO: Make sure obs_list has same first param dim in all elements
    def run_batched(self, obs_list: List[Observable]):
        print("Running with batch size {} for states, {} for observables and {} for samples".format(self.state_batch_size, 
                                                                                                    self.bs_obs,
                                                                                                    self.bs_N))
        N_obs = obs_list[0].params.shape[0]
        if self.bs_obs is None:
            self.bs_obs = N_obs

        assert self.N % self.bs_N == 0
        assert self.N_state_reps % self.state_batch_size == 0
        assert N_obs % self.bs_obs == 0
        for n in self.ns:
            self.estimator(n, self.N).validate_Ns(self.Ns)

        N_batches = self.N//self.bs_N
        N_state_batches = self.N_state_reps//self.state_batch_size
        # TODO: change so that batch size is always divisible
        obs_cls = type(obs_list[0])
        N_obs_batches = N_obs//self.bs_obs
        self.key, run_key = random.split(self.key)
        run_keys = list(random.split(run_key, len(self.ns)))

        extra_static_names = tuple(self.shadow_args(self.ns[0]).keys())
        @partial(jit, static_argnames=("n","N","N_state_reps","device","simulator") + extra_static_names)
        def init_shadow_jit(key: Array, n: int, N: int, N_state_reps: int=1, 
                sample_idx_range: Array=jnp.array([], dtype=jnp.int32),
               estimator: Estimator=Estimator(), 
               device: str= "lightning.qubit",
               simulator: str="pennylane",
               **kwargs):
            return self.shadow_cls.init(key=key, n=n, N=N, N_state_reps=N_state_reps, 
                                        sample_idx_range=sample_idx_range, estimator=estimator,
                                        device=device, simulator=simulator, **kwargs)

        @partial(jit, static_argnums=(1,2,3,4))
        def init_state_jit(key: Array, N_state: int, n: int,
                           *args, **kwargs):
            return self.state_cls.init_random(key=key, N_state=N_state, n=n,
                                              *args, **kwargs)

        # this function is called in tree.map
        def est_props(observable, n, key):
            key_shad, key_state, key_samp = random.split(key, 3)
            keys_shad = random.split(
                                        key_shad, (N_state_batches, N_batches)
                                    )
            keys_state = random.split(
                                        key_state, N_state_batches
                                    )
            keys_samp = random.split(
                                        key_samp, (N_state_batches, N_batches)
                                    )
            pred = jnp.zeros((self.N_state_reps, N_obs, self.Ns.shape[0]),
                             dtype=jnp.float32)
            gt = jnp.zeros((self.N_state_reps, N_obs),
                           dtype=jnp.float32)
            params = pred, gt

            @jit
            def loop_states(i, params):
                pred_vec, gt_vec = params

                # ONCE per state batch
                state = init_state_jit(
                    keys_state[i],
                    self.state_batch_size,
                    n,
                )

                # Ground truth depends only on the state and observables, but
                # the batched path used to leave gt_vec at its zero
                # initialization.  A one-shot shadow is sufficient to reuse
                # the common ground-truth circuit without allocating another
                # full sample batch.
                gt_shadow = init_shadow_jit(
                    keys_shad[i, 0],
                    n,
                    1,
                    self.state_batch_size,
                    self.gate_indices(n),
                    self.estimator(n, 1),
                    **self.shadow_args(n),
                )

                gt_state_batch = jnp.zeros(
                    (self.state_batch_size, N_obs),
                    dtype=jnp.float32,
                )

                @jit
                def loop_gt_obs(j, gt_state_batch):
                    obs = obs_cls.init(
                        dynamic_slice_in_dim(
                            observable.params,
                            j * self.bs_obs,
                            self.bs_obs,
                            axis=0,
                        ),
                        None,
                    )

                    ground_truth = jnp.asarray(
                        gt_shadow.ground_truth(state, obs),
                        dtype=jnp.float32,
                    )
                    return dynamic_update_slice_in_dim(
                        gt_state_batch,
                        ground_truth,
                        j * self.bs_obs,
                        axis=1,
                    )

                gt_state_batch = fori_loop(
                    0,
                    N_obs_batches,
                    loop_gt_obs,
                    gt_state_batch,
                )

                estimator = self.estimator(n, self.N)
                estimator_state = estimator.init_state(
                    (self.state_batch_size, N_obs), self.Ns
                )

                @jit
                def loop_N(k, estimator_state):

                    # ONCE per sample batch
                    shadow = init_shadow_jit(
                        keys_shad[i, k],
                        n,
                        self.bs_N,
                        self.state_batch_size,
                        self.gate_indices(n),
                        self.estimator(n, self.bs_N),
                        **self.shadow_args(n),
                    )

                    shadow = shadow.sample(
                        keys_samp[i, k],
                        state,
                    )

                    shadow = shadow.create_snapshots()

                    weak_batch = jnp.zeros(
                        (self.state_batch_size, N_obs, self.bs_N),
                        dtype=jnp.float32,
                    )

                    def loop_obs(j, weak_batch):
                        obs = obs_cls.init(
                            dynamic_slice_in_dim(
                                observable.params,
                                j * self.bs_obs,
                                self.bs_obs,
                                axis=0,
                            ),
                            None,
                        )

                        props = jnp.asarray(
                            shadow.estimate_weak_properties(obs),
                            dtype=jnp.float32,
                        )

                        weak_batch = dynamic_update_slice_in_dim(
                            weak_batch,
                            props,
                            j * self.bs_obs,
                            axis=1,
                        )

                        return weak_batch

                    weak_batch = fori_loop(
                        0,
                        N_obs_batches,
                        loop_obs,
                        weak_batch,
                    )

                    return estimator.update(estimator_state, weak_batch, self.Ns)

                if self.verbose:
                    loop_N = loop_tqdm(
                        N_batches,
                        print_rate=max(1, N_batches//20),
                        desc="Sample batches",
                    )(loop_N)

                estimator_state = fori_loop(
                    0,
                    N_batches,
                    loop_N,
                    estimator_state,
                )

                preds_state_batch = estimator.finalize(estimator_state, self.Ns)
                pred_vec = dynamic_update_slice_in_dim(
                    pred_vec,
                    preds_state_batch,
                    i * self.state_batch_size,
                    axis=0,
                )

                gt_vec = dynamic_update_slice_in_dim(
                    gt_vec,
                    gt_state_batch,
                    i * self.state_batch_size,
                    axis=0,
                )

                return pred_vec, gt_vec

            pred, gt = fori_loop(0, N_state_batches, loop_states, params)

            gt = jnp.stack([gt for _ in range(self.Ns.shape[0])], axis=-1)

            return pred, gt

        leave_fun = lambda x: isinstance(x, (int,Observable))
        est_all_props = jit(lambda: tree.map(est_props, obs_list, self.ns, run_keys,
                                             is_leaf=leave_fun))

        return est_all_props()


    def estimate_full(self, obs_list):
        leave_fun = lambda x: isinstance(x, (self.shadow_cls, self.state_cls, Observable))
        @partial(jit, static_argnums=((4,)))
        def compute_results(shadows, states, obs_list, Ns, batch_size):
            snap_shot_fun = lambda shadow: shadow.create_snapshots()
            shadows = tree.map(snap_shot_fun, shadows, is_leaf=leave_fun)
            pred_fun = jit(lambda shadow, observables, Ns: shadow.estimate_properties(observables, Ns, batch_size))

            gt_fun = jit(lambda shadow, states, obs: shadow.ground_truth(states, obs))
        
            preds = jnp.asarray(
                tree.map(
                    lambda shadow, observables: pred_fun(
                        shadow, observables, Ns
                    ),
                    shadows,
                    obs_list,
                    is_leaf=leave_fun,
                ),
                dtype=jnp.float32,
            )
            gt = jnp.asarray(
                tree.map(
                    gt_fun,
                    shadows,
                    states,
                    obs_list,
                    is_leaf=leave_fun,
                ),
                dtype=jnp.float32,
            )
            gt = jnp.stack([gt for _ in range(Ns.shape[0])], axis=-1)
            return preds, gt
        print("Compiled")
        return compute_results(self.shadows, 
                                self.states, 
                                obs_list, 
                                self.Ns,
                                self.state_batch_size)

    def estimate_gp(self, obs_list):
        gts = []
        preds = []
        for obs, shadow, states in zip(obs_list, self.shadows, self.states):
            pred = shadow.estimate_properties(obs, self.Ns)
            gtru = shadow.ground_truth(states, obs)
            gtruth = jnp.stack([gtru for _ in range(self.Ns.shape[0])], axis=-1)
            preds.append(pred)
            gts.append(gtruth)

        return jnp.stack(gts), jnp.stack(preds)

    def save_results(self, path: str):
        res = np.stack([self.gt, self.preds])
        np.save(path, res)

    def load_results(self, path: str):
        gt, preds = np.load(path)
        self.gt = jnp.asarray(gt, dtype=jnp.float32)
        self.preds = jnp.asarray(preds, dtype=jnp.float32)

    def plot(self, mode="data_scaling", path_save=None, intermediate_plot=False):
        # TODO: Change this
        if mode == "data_scaling":
            for i in range(self.observables_list[0].params.shape[0]):
                params = self.observables_list[0].params[i]
                for j, n in enumerate(self.ns):
                    plt.plot(np.asarray(self.Ns), jnp.mean(jnp.abs(self.preds[j,:,i,:]-self.gt[j,:,i,:]), axis=0), 
                            label=f"obs: {PauliObservable.obs_string(params)}, n: {n}")
        else:
            for i in range(self.observables_list[0].params.shape[0]):
                params = self.observables_list[0].params[i]
                for j, N in enumerate(self.Ns):
                    plt.plot(np.array(self.ns), np.abs(self.preds[j,:,i]-self.gt[j,:,i]), 
                            label=f"obs: {PauliObservable.obs_string(params)}, N: {N}")
        
        if not intermediate_plot:
            plt.legend()
            plt.show()

        if path_save is not None:
            plt.savefig(path_save)

def test():
    ns = list(range(10,11))
    Ns = [2**k for k in range(5,13)]
    # obs_list = [[PauliObservable(O*n) for O in ["X","Y","Z"]] for n in ns]
    obs_list = [PauliObservable.init_random(random.PRNGKey(34251), n=n, N=100) for n in ns]
    experiment = ShadowScalingExperiment(SEEQSTShadow, HRState, 
                                         N_list=Ns, n_list=ns, obs_lists=obs_list, N_state_reps=500,
                                         gate_indices=lambda n: jnp.array([0,2], dtype=jnp.int32))
    experiment.run()
    experiment.plot()

if __name__ == "__main__":
    test()
