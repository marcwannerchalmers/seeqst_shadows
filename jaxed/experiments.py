from typing import List, Type
import numpy as np
from jax import numpy as jnp
from jax import random, vmap, jit
from jax import tree
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import partial
import time

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxed.shadow import PauliShadow, CliffordShadow, SEEQSTShadow, Shadow
from jaxed.tools.estimator import Estimator
from jaxed.tools.observable import PauliObservable, Observable
from jaxed.tools.state import HRState, State

# TODO: Save the outcomes and load them if the path exists, printing what it ended up doing.
class ShadowScalingExperiment:
    def __init__(self, shadow_cls: Type[Shadow], state_cls: Type[State], shadow_args={}, state_args={}, state_name=None,
                  N_list=[], n_list=[], obs_lists:List[PauliObservable]=[],
                  gate_indices=lambda n: [0,2], estimator=lambda n, N: Estimator(),
                  N_state_reps: int=1, key=random.PRNGKey(12345), 
                  verbose=False, state_batch_size=3) -> None:
        self.shadow_cls = shadow_cls
        self.state_cls = state_cls
        self.shadow_args = shadow_args
        self.state_args = state_args
        self.Ns = jnp.array(N_list)
        self.ns = n_list
        self.N_state_reps = N_state_reps
        self.observables_list = []
        if len(obs_lists) == len(self.ns):
            self.observables_list = obs_lists

        self.preds = jnp.zeros((len(self.Ns), len(self.ns), self.observables_list[0].params.shape[0]))
        self.gt = jnp.zeros_like(self.preds)
        self.gate_indices = gate_indices
        self.estimator = estimator
        self.state_name = state_name
        self.times = {"Create Shadow": [], "Predict": [], "Compute GT": [], "Total step": []}
        self.verbose = verbose
        self.state_batch_size = state_batch_size
        self.init(key)

    def init(self, key):
        keys1, keys2 = random.split(key, (2,len(self.ns)))
        self.states = [self.state_cls.init_random(keys1[i], self.N_state_reps, n)  
                                for i, n in enumerate(self.ns)]
        N = int(jnp.max(self.Ns))

        self.shadows = [self.shadow_cls.init(keys2[i], n, N, self.N_state_reps,
                                              self.gate_indices(n), self.estimator(n,N),
                                              **self.shadow_args)
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
            leave_fun = lambda x: isinstance(x, (self.shadow_cls, self.state_cls, Observable))
            # sampling
            # sample_fun = lambda shadow, states: shadow.sample(states)
            """replace_fun = lambda shadow, outcomes: shadow.replace(outcomes=outcomes)
            outcomes = tree.map(sample_fun, self.shadows, self.states) 
            self.shadows = tree.map(replace_fun, self.shadows, outcomes)"""
            # self.shadows = tree.map(sample_fun, self.shadows, self.states,
                                    # is_leaf=leave_fun)
            @jit
            def sample_shadows(shadows, states):
                sample_fun = lambda shadow, states: shadow.sample(states)
                return tree.map(sample_fun, shadows, states,
                                    is_leaf=leave_fun)
            print("Sampling {} from {}-{} qubits".format(str(type(self.states[0])).split(".")[-1].split("\'")[0], 
                                                                     min(self.ns), 
                                                                     max(self.ns)))
            start = time.time()
            self.shadows = sample_shadows(self.shadows, self.states)
            end = time.time()
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
            self.preds, self.gt = compute_results(self.shadows, 
                                                  self.states, 
                                                  obs_list, 
                                                  self.Ns,
                                                  self.state_batch_size)
            print(self.preds.shape, self.gt.shape)
            end = time.time()

            print("Creating {} snapshots and processing them for {} observables each took {} s".format(len(self.ns)*self.N_state_reps*max(self.Ns),
                                                  self.observables_list[0].params.shape[0]*len(self.ns)*self.N_state_reps,
                                                  end-start))


        if save_state:
            self.save_results(path_save)

    def save_results(self, path: str):
        res = np.stack([self.gt, self.preds])
        np.savetxt(path, res)

    def load_results(self, path: str):
        self.gt, self.preds = np.loadtxt(path)

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

class Experiments:
    def __init__(self, scaling_experiments: List[ShadowScalingExperiment],
                 name_dict = None) -> None:
        self.experiments = scaling_experiments
        for exp in scaling_experiments:
            exp.run() 
        self.data = self.get_dataframe()

    def get_dataframe(self):
        data = {"Shadow model": [], "Observable": [], "Ground truth": [],
                "Prediction": [], "Error": [], "N": [], "n": [], "State": []}
        for experiment in self.experiments:        
            # print(experiment.gt.shape, experiment.preds.shape)  
            for j, (observables, n) in enumerate(zip(experiment.observables_list, experiment.ns)):
                for k, obs in enumerate(observables):
                    for i, N in enumerate(experiment.Ns):
                        data["Shadow model"].append(re.search(r"\.(.*?)\'", str(experiment.shadow_cls))[1])
                        data["Observable"].append(obs.get_name())
                        data["Ground truth"].append(experiment.gt[i,j,k])
                        data["Prediction"].append(experiment.preds[i,j,k])
                        data["Error"].append(abs(experiment.gt[i,j,k]-experiment.preds[i,j,k]))
                        data["N"].append(N)
                        data["n"].append(n)
                        state_name = str(experiment.state_cls) if experiment.state_name is None else experiment.state_name
                        data["State"].append(state_name)

        return pd.DataFrame(data)

    # partial function with self.data fixed 
    def plot(self, index_list=[], path_save=None, **lineplot_args):
        data = np.asarray(self.data)
        if len(index_list) > 0:
            data = data[index_list]
        sns.set_style('whitegrid')
        res = sns.lineplot(data, **lineplot_args)
        res.set(xscale='log')
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