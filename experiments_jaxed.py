from jaxed.tools.observable import PauliObservable
from jaxed.tools.state import HRState, HChainGS
import pennylane as qp
from jaxed.shadow import SEEQSTShadow, PauliShadow, CliffordShadow
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
from jaxed.tools.estimator import Estimator, MedianOfMeans
from jaxed.experiments import ShadowScalingExperiment
import seaborn as sns
import pandas as pd
from functools import partial
import re
import time
import jax
from jax import random
from GPshadow import GPShadow

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
                for l in range(experiment.N_state_reps):
                    for k in range(observables.params.shape[0]):
                        for i, N in enumerate(experiment.Ns):
                            model_name = re.search(r"\.(.*?)\'", str(experiment.shadow_cls))[1]
                            if experiment.shadow_cls is GPShadow:
                                model_name = experiment.shadows[0].cfg["kernel_type"]  
                            data["Shadow model"].append(model_name)
                            data["Observable"].append(observables.obs_string(observables.params[k]))
                            data["Ground truth"].append(experiment.gt[j,l,k,i])
                            data["Prediction"].append(experiment.preds[j,l,k,i])
                            data["Error"].append(abs(experiment.gt[j,l,k,i]-experiment.preds[j,l,k,i]))
                            data["N"].append(N)
                            data["n"].append(n)
                            state_name = str(experiment.state_cls) if experiment.state_name is None else experiment.state_name
                            data["State"].append(state_name + "_" + str(l))

        for key, val in data.items():
            data[key] = np.array(val)
        return pd.DataFrame(data)

    # TODO: Check if hue can be done over multiple arguments
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

    def plot_avg_var(self, path_save=None):
        df = self.data
        print(df)
        # Variance across predictions within each experiment
        df_var = (
            df.groupby(["Shadow model", "N", "n", "State"])["Error"]
            .var()
            .reset_index(name="variance")
        )

        df_mean_err = (
                df.groupby(["Shadow model", "N", "n", "State"])["Error"]
                .mean()
                .reset_index(name="mean")
            )

        df_mean_err = (
                        df_mean_err.groupby(["Shadow model", "N", "n"])["mean"]
                        .mean()
                        .reset_index(name="mean abs Error")
                    )

        # Average variance across experiments
        avg_var = (
            df_var.groupby(["Shadow model", "N", "n"])["variance"]
            .mean()
            .reset_index()
        )

        plot_df = df_mean_err.merge(avg_var, on=["Shadow model", "N", "n"])
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        res = sns.lineplot(plot_df, x="N", y="mean abs Error", hue="Shadow model", ax=ax)
        # res.set(xscale='log')
        plot_df["std"] = np.sqrt(plot_df["variance"])
        for method, g in plot_df.groupby("Shadow model"):
            ax.fill_between(
                g["N"].to_numpy(),
                (g["mean abs Error"] - g["std"]).to_numpy(),
                (g["mean abs Error"] + g["std"]).to_numpy(),
                alpha=0.05,
            )

        # plt.show()
        ax.set_ylim(0, plot_df["mean abs Error"].max()+0.1)
        if path_save is not None:
            plt.savefig(path_save)



def test_multiple_hom_paulis(n: int):
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 15)]
    N_obs = 50
    reps = 5
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2,3], n=n, N=N_obs) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQST", "results/Pauli", "results/Clifford"]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    kernels = ["Hamming", "SeparatedHamming", "PauliProduct"]
    paths_GP = ["results/Hamming", "results/SeparatedHamming", "results/PauliProduct"]
    experiments_list = []
    for i, ker in enumerate(kernels):
        shadow_args = {"cfg":{"kernel_type": ker, "kernel_args": {},
                            "share_parameters": True}}
        exp = ShadowScalingExperiment(GPShadow, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=paths_GP[i],k_local=n)
        experiments_list.append(exp)
        
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=data_paths[i], 
                                         state_batch_size=1,k_local=n
                                         )
        experiments_list.append(exp)
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)))

def test_multiple_hom_paulis_klocal(n,k):
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 15)]
    N_obs = 50
    reps = 5
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2,3], n=n, N=N_obs,k_local=k) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQST", "results/Pauli", "results/Clifford"]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    kernels = ["Hamming", "SeparatedHamming", "PauliProduct"]
    paths_GP = ["results/Hamming", "results/SeparatedHamming", "results/PauliProduct"]
    experiments_list = []
    for i, ker in enumerate(kernels):
        shadow_args = {"cfg":{"kernel_type": ker, "kernel_args": {},
                            "share_parameters": True}}
        exp = ShadowScalingExperiment(GPShadow, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=paths_GP[i],k_local=k)
        experiments_list.append(exp)
        
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                         N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                         verbose=True, key=key2, path_save=data_paths[i], 
                                         state_batch_size=1,k_local=k
                                         )
        experiments_list.append(exp)
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_k{}_Nobs{}_reps{}_N{}.pdf".format(n,k,N_obs,reps,max(Ns)))

def XY_combos(n):
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 17)]
    N_obs = 50
    reps = 1
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQSTXY", "results/PauliXY", "results/CliffordXY"]
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiments_list=[]
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i], 
                                            state_batch_size=1
                                            )
        experiments_list.append(exp)
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)))

if __name__ == "__main__":
    """test_multiple_hom_paulis(4)
    test_multiple_hom_paulis_klocal(4,2)
    test_multiple_hom_paulis(5)
    test_multiple_hom_paulis_klocal(5,2)
    test_multiple_hom_paulis(10)
    test_multiple_hom_paulis_klocal(10,2)"""
    XY_combos(10)