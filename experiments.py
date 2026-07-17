from tools.observable import PauliObservable
from tools.state import HRState
import pennylane as qp
from shadow import SEEQSTShadow, PauliShadow, CliffordShadow
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from tools.estimator import Estimator
import seaborn as sns
import pandas as pd

# TODO: make such that observables are tied to number of qubits
class ShadowScalingExperiment:
    def __init__(self, shadow_cls, state_cls, shadow_args={}, state_args = {}, state_name=None,
                  N_list=[], n_list=[], obs_lists:List[List[PauliObservable]]=[],
                  gate_indices=lambda n: [0,2**n], estimator=lambda n: Estimator()) -> None:
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

    def step(self, N, n, observables):
        # shadow = SEEQSTShadow(HRState(n), [0, 2**n], full_setting=False)
        shadow = self.shadow_cls(self.state_cls(n, **self.state_args), 
                                 self.gate_indices(n), self.estimator(n), **self.shadow_args)
        shadow.create(N)
        return shadow.predict(observables), shadow.ground_truth(observables)
    
    def run(self, path_save = None):
        save_state = False
        if path_save is not None:
            if os.path.isfile(path_save):
                self.load_results(path_save)
            else:
                save_state = True

        for i, N in enumerate(self.Ns):
            for j, (n, observables) in enumerate(zip(self.ns, self.observables_list)):
                self.preds[i,j], self.gt[i,j] = self.step(N, n, observables)
                print(f"N: {N}, n: {n}, first observable: {observables[0].get_name()}, \
                      GT: {self.gt[i,j]}, Pred: {self.preds[i,j]}")

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
        self.data = self.get_dataframe()

    def get_dataframe(self):
        data = {"Shadow model": [], "Observable": [], "Ground truth": [],
                "Prediction: ": [], "Error": [], "N": [], "n": [], "State": []}
        for experiment in self.experiments:          
            for i, (observables, n) in enumerate(zip(experiment.observables_list, experiment.ns)):
                for k, obs in enumerate(observables):
                    for j, N in enumerate(experiment.Ns):
                        data["Shadow model"].append(str(experiment.shadow_cls))
                        data["Observable"].append(obs.get_name())
                        data["Ground truth"].append(experiment.gt[i,j,k])
                        data["Prediction"].append(experiment.preds[i,j,k])
                        data["Error"].append(abs(experiment.gt[i,j,k]-experiment.preds[i,j,k]))
                        data["N"].append(N)
                        data["n"].append(n)
                        state_name = str(experiment.state_cls) if experiment.state_name is None else experiment.state_name
                        data["State"].append(state_name)

        return pd.DataFrame(data)
    
    def plot(self, x, y, hue=None, index_list=[], path_save=None):
        data = self.data
        if len(index_list) > 0:
            data = data[index_list]
        
        sns.lineplot(data, x=x, y=y, hue=hue)
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

def test_XY_combos():
    # TODO: Code up experiment for only the antidiagonal block of SEEQST shadow
    ns = list(range(4,5))
    Ns = [2**k for k in range(12, 13)]
    reps = 2
    state_reps = 2
    sample_idx_list = [[2*np.ones((n,)), 4*np.ones((n,))] for n in ns]
    obs_list = [[PauliObservable(n, sample_indices) for _ in range(reps)] for n, sample_indices in 
                 zip(ns, sample_idx_list)]
    state_names = ["HR_state_"+str(i) for i in range(state_reps)]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiments_list = []
    for shadow_cls in shadow_classes:
        experiments_list.extend([ShadowScalingExperiment(shadow_cls, HRState, {"full_setting": False}, state_names[i],
                                         Ns, ns, obs_list) for i in range(state_reps)])
    experiments = Experiments(experiments_list)
    # TODO: figure out how to plot this
    # experiments.plot()

def main():
    test_homogenous_paulis()


if __name__ == "__main__":
    main()