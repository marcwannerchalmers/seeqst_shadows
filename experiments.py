from tools.observable import PauliObservable
from tools.state import HRState, HChainGS
import pennylane as qp
from shadow import SEEQSTShadow, PauliShadow, CliffordShadow
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from tools.estimator import Estimator, MedianOfMeans
import seaborn as sns
import pandas as pd
from functools import partial
import re
import time

# TODO: make such that observables are tied to number of qubits
class ShadowScalingExperiment:
    def __init__(self, shadow_cls, state_cls, shadow_args={}, state_args={}, state_name=None,
                  N_list=[], n_list=[], obs_lists:List[List[PauliObservable]]=[],
                  gate_indices=lambda n: [0,2**n], estimator=lambda n, N: Estimator(),
                  verbose=False) -> None:
        self.shadow_cls = shadow_cls
        self.state_cls = state_cls
        self.shadow_args = shadow_args
        self.state_args = state_args
        self.Ns = N_list
        self.ns = n_list
        self.observables_list = []
        if len(obs_lists) == len(self.ns):
            self.observables_list = obs_lists

        self.preds = np.zeros((len(self.Ns), len(self.ns), len(self.observables_list[0])))
        self.gt = np.zeros_like(self.preds)
        self.gate_indices = gate_indices
        self.estimator = estimator
        self.state_name = state_name
        self.times = {"Create Shadow": [], "Predict": [], "Compute GT": [], "Total step": []}
        self.verbose = verbose

    def step(self, N, n, observables):
        # shadow = SEEQSTShadow(HRState(n), [0, 2**n], full_setting=False)
        shadow = self.shadow_cls(self.state_cls(n, **self.state_args), 
                                 self.gate_indices(n), self.estimator(n, N), **self.shadow_args)
        start = time.time()
        shadow.create(N)
        created = time.time()
        pred = shadow.predict(observables)
        predicted = time.time()
        gt = shadow.ground_truth(observables)
        end = time.time()
        self.times["Create Shadow"].append(created - start)
        self.times["Predict"].append(predicted - created)
        self.times["Compute GT"].append(end - predicted)
        return pred, gt
    
    def run(self, path_save = None):
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
            # This block can potentially be jaxed
            for i, N in enumerate(self.Ns):
                for j, (n, observables) in enumerate(zip(self.ns, self.observables_list)):
                    start = time.time()
                    self.preds[i,j], self.gt[i,j] = self.step(N, n, observables)
                    end = time.time()
                    self.times["Total step"].append(end - start)
                    if self.verbose:
                        print(f"N: {N}, n: {n}, first observable: {observables[0].get_name()}, \
                            GT: {self.gt[i,j]}, Pred: {self.preds[i,j]}")
                        print("Time taken: " + "".join(["{}: {} s, ".format(key, value[-1]) 
                                                   for key, value in self.times.items()]))
                    
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

# TODO: Add saving function if loading data takes too long
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
        data = self.data
        if len(index_list) > 0:
            data = data[index_list]
        sns.set_style('whitegrid')
        res = sns.lineplot(data, **lineplot_args)
        res.set(xscale='log')
        plt.show()
        if path_save is not None:
            plt.savefig(path_save)
    



# TODO: Run experiments for the three shadows
def test_homogenous_paulis():
    ns = list(range(4,5))
    Ns = [2**k for k in range(12, 13)]
    # obs_list = [[PauliObservable(O*n) for O in ["X","Y","Z"]] for n in ns]
    obs_list = [[PauliObservable(O*n) for O in ["X"]] for n in ns]
    experiment = ShadowScalingExperiment(SEEQSTShadow, HRState, {"full_setting": False}, 
                                         N_list=Ns, n_list=ns, obs_lists=obs_list)
    experiment.run()
    # experiment.plot()

def test_multiple_hom_paulis():
    ns = list(range(10,11))
    Ns = [2**k for k in range(7, 15)]
    reps = 2
    state_reps = 1
    obs_lists = [[PauliObservable(O*2 + "I"*(n-2), name=O) for O in ["X", "Y", "Z"]] for n in ns]
    state_names = ["HR_state_"+str(i) for i in range(state_reps)]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiments_list = []
    for shadow_cls in shadow_classes:
        shadow_args = {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exps = [ShadowScalingExperiment(shadow_cls, HChainGS, shadow_args=shadow_args, state_name=state_names[i],
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, verbose=True)#,
                                         #estimator=lambda n, N: MedianOfMeans(np.sqrt(N))) 
                                         for i in range(state_reps)]
        experiments_list.extend(exps)
    experiments = Experiments(experiments_list)
    experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))


def test_XY_combos():
    # TODO: Code up experiment for only the antidiagonal block of SEEQST shadow
    ns = list(range(10,11))
    Ns = [2**k for k in range(12, 13)]
    reps = 2
    state_reps = 2
    sample_idx_list = [[2*np.ones((n,)), 4*np.ones((n,))] for n in ns]
    obs_lists = [[PauliObservable(n, sample_indices) for _ in range(reps)] for n, sample_indices in 
                 zip(ns, sample_idx_list)]
    state_names = ["HR_state_"+str(i) for i in range(state_reps)]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiments_list = []
    for shadow_cls in shadow_classes:
        exps = [ShadowScalingExperiment(shadow_cls, HRState, shadow_args={"full_setting": False}, state_name=state_names[i],
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists) for i in range(state_reps)]
        experiments_list.extend(exps)
    experiments = Experiments(experiments_list)
    # TODO: figure out how to plot this
    # experiments.plot()

def testGP():
    pass

def main():
    test_multiple_hom_paulis()


if __name__ == "__main__":
    main()