from typing import List, Type
import numpy as np
import jax
from jax import numpy as jnp
from jax import random, vmap, jit, Array
from jax import tree
from jax.lax import fori_loop, cond, dynamic_slice_in_dim, dynamic_update_slice_in_dim, \
                    dynamic_update_slice, dynamic_slice
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import partial, lru_cache
import time

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxed.shadow import PauliShadow, CliffordShadow, SEEQSTShadow, Shadow
from jaxed.tools.estimator import Estimator
from jaxed.tools.observable import PauliObservable, Observable
from jaxed.tools.state import HRState, State
from GPshadow import GPShadow

# TODO: Save the outcomes and load them if the path exists, printing what it ended up doing.
class ShadowScalingExperiment:
    def __init__(self, shadow_cls: Type[Shadow], state_cls: Type[State], shadow_args={}, state_args={}, state_name=None,
                  N_list=[], n_list=[], obs_lists:List[PauliObservable]=[],
                  gate_indices=lambda n: jnp.array([0,2]), estimator=lambda n, N: Estimator(),
                  N_state_reps: int=1, key=random.PRNGKey(12345), 
                  verbose=False, state_batch_size=3, path_save=None, k_local=None,
                  batch_size_obs=None,
                  batch_size_N=None,
                  experiment_name=None) -> None:
        self.shadow_cls = shadow_cls
        self.state_cls = state_cls
        self.shadow_args = shadow_args
        self.state_args = state_args
        self.Ns = jnp.array(N_list, dtype=int)
        self.N = int(jnp.max(self.Ns))
        self.ns = n_list
        self.N_state_reps = N_state_reps
        self.observables_list = []
        if len(obs_lists) == len(self.ns):
            self.observables_list = obs_lists

        self.k_local = k_local

        self.preds = jnp.zeros((len(self.Ns), len(self.ns), self.observables_list[0].params.shape[0]))
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
                f"{path_save}_nmax{max(self.ns)}_N{max(self.Ns)}"
                f"_reps{self.N_state_reps}_k{self.k_local}{batch_suffix}"
            )
        self.init(key)
        self.key = random.split(key)[1]
        self.experiment_name = experiment_name

    def init(self, key):
        keys1, keys2 = random.split(key, (2,len(self.ns)))
        self.states = [self.state_cls.init_random(keys1[i], self.N_state_reps, n)  
                                for i, n in enumerate(self.ns)]
        N = self.N

        self.shadows = [self.shadow_cls.init(keys2[i], n, N, self.N_state_reps,
                                              self.gate_indices(n), self.estimator(n,N),
                                              **self.shadow_args)
                                for i, n in enumerate(self.ns)]

    
    def run(self):
        save_state = False
        experiments_loaded = False
        if self.path_save is not None:
            if os.path.isfile(self.path_save+".npy"):
                self.load_results(self.path_save+".npy")
                experiments_loaded = True
            else:
                if self.verbose:
                    print("Experiments will be saved as {}".format(self.path_save))
                save_state = True

        if not experiments_loaded:
            if self.use_batched:
                start = time.time()
                res = self.run_batched(self.observables_list)
                preds, gt = zip(*res)
                self.preds = jnp.stack(preds, axis=0)
                self.gt = jnp.stack(gt, axis=0)
                end = time.time()
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
                start = time.time()
                self.shadows = sample_shadows(self.shadows, self.states)
                end = time.time()
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
                if self.shadow_cls is GPShadow:
                    self.gt, self.preds = self.estimate_gp(obs_list)
                else: 
                    self.preds, self.gt = self.estimate_full(obs_list) 
                end = time.time()

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

        N_batches = self.N//self.bs_N
        N_state_batches = self.N_state_reps//self.state_batch_size
        # TODO: change so that batch size is always divisible
        obs_cls = type(obs_list[0])
        N_obs_batches = N_obs//self.bs_obs

        extra_static_names = tuple(self.shadow_args.keys())
        @partial(jit, static_argnames=("n","N","N_state_reps","device","simulator") + extra_static_names)
        def init_shadow_jit(key: Array, n: int, N: int, N_state_reps: int=1, 
                sample_idx_range: Array=jnp.array([]), 
               estimator: Estimator=Estimator(), 
               device: str= "lightning.qubit",
               simulator: str="statevector",
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
        def est_props(observable, n):
            self.key, key_shad, key_state, key_samp = random.split(self.key, 4)
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

            def loop_states(i, params):
                pred_vec, gt_vec = params

                # ONCE per state batch
                state = init_state_jit(
                    keys_state[i],
                    self.state_batch_size,
                    n,
                )

                weak_pred_vec = jnp.zeros(
                    (self.state_batch_size, N_obs, self.N),
                    dtype=jnp.float32,
                )

                def loop_N(k, weak_pred_vec):

                    # ONCE per sample batch
                    shadow = init_shadow_jit(
                        keys_shad[i, k],
                        n,
                        self.bs_N,
                        self.state_batch_size,
                        self.gate_indices(n),
                        self.estimator(n, self.bs_N),
                        **self.shadow_args,
                    )

                    shadow = shadow.sample(
                        keys_samp[i, k],
                        state,
                    )

                    shadow = shadow.create_snapshots()

                    def loop_obs(j, weak_pred_vec):
                        obs = obs_cls.init(
                            dynamic_slice_in_dim(
                                observable.params,
                                j * self.bs_obs,
                                self.bs_obs,
                                axis=0,
                            ),
                            None,
                        )

                        props = shadow.estimate_weak_properties(obs)
                        props = jnp.asarray(props, dtype=weak_pred_vec.dtype)

                        weak_pred_vec = dynamic_update_slice(
                            weak_pred_vec,
                            props,
                            (
                                0,
                                j * self.bs_obs,
                                k * self.bs_N,
                            ),
                        )

                        return weak_pred_vec

                    weak_pred_vec = fori_loop(
                        0,
                        N_obs_batches,
                        loop_obs,
                        weak_pred_vec,
                    )

                    return weak_pred_vec

                weak_pred_vec = fori_loop(
                    0,
                    N_batches,
                    loop_N,
                    weak_pred_vec,
                )

                estimator = self.estimator(n, self.N)
                pred_fun = lambda estimator, props, N: estimator(
                    props,
                    N,
                )     
                preds_state_batch = vmap(
                    pred_fun,
                    in_axes=(None, None, 0),
                    out_axes=2,
                )(
                    estimator,
                    weak_pred_vec,
                    self.Ns,
                )
                pred_vec = dynamic_update_slice_in_dim(
                    pred_vec,
                    preds_state_batch,
                    i * self.state_batch_size,
                    axis=0,
                )

                return pred_vec, gt_vec

            pred, gt = fori_loop(0, N_state_batches, loop_states, params)

            gt = jnp.stack([gt for _ in range(self.Ns.shape[0])], axis=-1)

            return pred, gt

        leave_fun = lambda x: isinstance(x, (int,Observable))
        est_all_props = jit(lambda: tree.map(est_props, obs_list, self.ns,
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
        
            preds = jnp.array(tree.map(lambda shadow, observables: pred_fun(shadow, observables, Ns), 
                                        shadows, 
                                        obs_list, 
                                        is_leaf=leave_fun))
            gt = jnp.array(tree.map(gt_fun, shadows, states, obs_list, is_leaf=leave_fun))
            gt = jnp.stack([gt for _ in range(Ns.shape[0])], axis=-1)
            return preds, gt
        print("Compiled")
        start = time.time()
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
        self.gt, self.preds = np.load(path)

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
                                         gate_indices=lambda n: jnp.array([0,2]))
    experiment.run()
    experiment.plot()

if __name__ == "__main__":
    test()
