from typing import List, Type
import numpy as np
from jax import numpy as jnp
from jax import random, vmap
from jax import tree
import os
import matplotlib.pyplot as plt

from jaxed.shadow import PauliShadow, CliffordShadow, SEEQSTShadow, Shadow
from jaxed.tools.estimator import Estimator
from jaxed.tools.observable import PauliObservable, Observable
from jaxed.tools.state import HRState, State


class ShadowScalingExperiment:
    def __init__(self, shadow_cls: Type[Shadow], state_cls: Type[State], shadow_args={}, state_args={}, state_name=None,
                  N_list=[], n_list=[], obs_lists:List[List[PauliObservable]]=[],
                  gate_indices=lambda n: [0,2**n], estimator=lambda n, N: Estimator(),
                  N_state_reps: int=1, key=random.PRNGKey(12345), 
                  verbose=False) -> None:
        self.shadow_cls = shadow_cls
        self.state_cls = state_cls
        self.shadow_args = shadow_args
        self.state_args = state_args
        self.Ns = N_list
        self.ns = n_list
        self.N_state_reps = N_state_reps
        self.observables_list = []
        if len(obs_lists) == len(self.ns):
            self.observables_list = obs_lists

        self.preds = jnp.zeros((len(self.Ns), len(self.ns), len(self.observables_list[0])))
        self.gt = jnp.zeros_like(self.preds)
        self.gate_indices = gate_indices
        self.estimator = estimator
        self.state_name = state_name
        self.times = {"Create Shadow": [], "Predict": [], "Compute GT": [], "Total step": []}
        self.verbose = verbose
        self.init(key)

    def init(self, key):
        keys1, keys2 = random.split(key, (2,len(self.ns),len(self.Ns)))
        self.states = [[self.state_cls.init_random(keys1[i,j], self.N_state_reps, n) 
                            for j, N in enumerate(self.ns)]
                                for i, n in enumerate(self.ns)]

        self.shadows = [[self.shadow_cls.init(keys2[i,j], n, N, self.N_state_reps,
                                              self.gate_indices(n), self.estimator(n,N),
                                              **self.shadow_args)
                            for j, N in enumerate(self.ns)]
                                for i, n in enumerate(self.ns)]
    
    def run(self, path_save=None):
        save_state = False
        experiments_loaded = False
        if path_save is not None:
            if os.path.isfile(path_save):
                self.load_results(path_save)
                experiments_loaded = True
            else:
                if self.verbose:
                    print("Experiments will be saved as {}".format(path_save))
                save_state = True

        if not experiments_loaded:
            # sampling
            sample_fun = lambda shadow, states: vmap(shadow.sample)(states)
            replace_fun = lambda shadow, outcomes: shadow.replace(outcomes=outcomes)
            outcomes = tree.map(sample_fun, self.shadows, self.states) 
            self.shadows = tree.map(replace_fun, self.shadows, outcomes)

            # prediction
            obs_list = self.observables_list
            # Adapt observable 
            # Case 1: Same observables for each N --> list of len(ns) observables
            if not isinstance(self.observables_list[0], list):
                obs_list = [[obs for _ in range(len(self.Ns))] for obs in self.observables_list]
            # Case 2: Same observables for each state 
            # --> shape_0 of params does not equal N_states for ALL observables
            case2_fun = lambda obs: obs.params.shape[0] == self.N_state_reps
            case2 = not jnp.array(tree.map(case2_fun, obs_list)).all()
            obs_axes = (None, 0) if case2 else (0,1)
            pred_fun = lambda shadow, obs: shadow.predict(obs)
            pred_fun = vmap(vmap(pred_fun, 
                                 in_axes=(None, obs_axes[0])), 
                                 in_axes=(0,obs_axes[1]))

            gt_fun = lambda shadow, obs: shadow.ground_truth(obs)
            gt_fun = vmap(vmap(gt_fun, 
                                 in_axes=(None, obs_axes[0])), 
                                 in_axes=(0,obs_axes[1]))
            
            self.preds = jnp.array(tree.map(gt_fun, self.shadows, obs_list))
            self.gts = jnp.array(tree.map(gt_fun, self.shadows, obs_list))
                    
        if save_state:
            self.save_results(path_save)

    def save_results(self, path: str):
        res = np.stack([self.gt, self.preds])
        np.savetxt(path, res)

    def load_results(self, path: str):
        self.gt, self.preds = np.loadtxt(path)

    def plot(self, mode="data_scaling", path_save=None, intermediate_plot=False):
        if mode == "data_scaling":
            for i, obs in enumerate(self.observables_list[0]):
                for j, n in enumerate(self.ns):
                    plt.plot(np.array(self.Ns), np.abs(self.preds[:,j,i]-self.gt[:,j,i]), 
                            label=f"obs: {obs.get_name()}, n: {n}")
        else:
            for i, obs in enumerate(self.observables_list[0]):
                for j, N in enumerate(self.Ns):
                    plt.plot(np.array(self.ns), np.abs(self.preds[j,:,i]-self.gt[j,:,i]), 
                            label=f"obs: {obs.get_name()}, N: {N}")
        
        if not intermediate_plot:
            plt.legend()
            plt.show()

        if path_save is not None:
            plt.savefig(path_save)