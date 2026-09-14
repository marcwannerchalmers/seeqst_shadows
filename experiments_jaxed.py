from jaxed.tools.observable import PauliObservable
from jaxed.tools.state import HRState, HChainGS
from jaxed.tools.distributions import binomial
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
from matplotlib.ticker import LogLocator, LogFormatterMathtext, ScalarFormatter
from scipy.stats import t

class Experiments:
    def __init__(self, scaling_experiments: List[ShadowScalingExperiment],
                 name_dict = None) -> None:
        self.experiments = scaling_experiments
        for exp in scaling_experiments:
            exp.run() 
        self.data = self.get_dataframe()

    def get_dataframe(self):
        data = {"Shadow model": [], "Observable": [], "Ground truth": [],
                "Prediction": [], "Error": [], "N": [], "n": [], "State": [],
                "experiment_name": []}
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
                            data["experiment_name"].append(experiment.experiment_name)

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

    def plot_avg_var(self, path_save=None, xlog=False, ylog=False):
        df = self.data

        # ------------------------------------------------------------
        # 1. Average over observables for each individual state
        #
        # This leaves one error value for every:
        #   method x N x n x state
        # ------------------------------------------------------------
        state_means = (
            df.groupby(
                ["Shadow model", "N", "n", "State", "experiment_name"]
            )["Error"]
            .mean()
            .reset_index(name="state_mean_error")
        )

        # ------------------------------------------------------------
        # 2. Compute mean and uncertainty across state repetitions
        # ------------------------------------------------------------
        plot_df = (
            state_means
            .groupby(
                ["Shadow model", "N", "n", "experiment_name"]
            )["state_mean_error"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Standard error of the mean
        plot_df["sem"] = (
            plot_df["std"] / np.sqrt(plot_df["count"])
        )

        # 95% Student-t confidence interval
        plot_df["tcrit"] = plot_df["count"].apply(
            lambda n: t.ppf(0.975, df=n - 1)
            if n > 1 else np.nan
        )

        plot_df["ci"] = (
            plot_df["tcrit"] * plot_df["sem"]
        )

        plot_df["ci_low"] = (
            plot_df["mean"] - plot_df["ci"]
        )

        plot_df["ci_high"] = (
            plot_df["mean"] + plot_df["ci"]
        )

        # ------------------------------------------------------------
        # 3. Create figure
        # ------------------------------------------------------------
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(
            figsize=(7.5, 5.5)
        )

        # Different markers for the different shadow methods
        sns.lineplot(
            data=plot_df,
            x="N",
            y="mean",
            hue="experiment_name",
            style="experiment_name",
            markers=True,
            dashes=False,
            markersize=7,
            linewidth=1.8,
            ax=ax,
        )

        # ------------------------------------------------------------
        # 4. Confidence intervals
        # ------------------------------------------------------------

        # On a logarithmic y-axis, values <= 0 cannot be plotted.
        if ylog:
            positive_values = plot_df.loc[
                plot_df["mean"] > 0,
                "mean",
            ]

            if len(positive_values) > 0:
                y_floor = positive_values.min() * 1e-3
            else:
                y_floor = 1e-12
        else:
            y_floor = None

        for method, g in plot_df.groupby(
            "experiment_name"
        ):
            # Make sure N is sorted before fill_between
            g = g.sort_values("N")

            x = g["N"].to_numpy()
            lower = g["ci_low"].to_numpy()
            upper = g["ci_high"].to_numpy()

            if ylog:
                # Only clip for plotting.
                # The actual confidence interval values remain unchanged.
                lower = np.maximum(
                    lower,
                    y_floor,
                )

            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.15,
            )

        # ------------------------------------------------------------
        # 5. X axis
        # ------------------------------------------------------------
        if xlog:
            # N consists of powers of 2 in your experiments
            ax.set_xscale(
                "log",
                base=2,
            )

            xvals = np.sort(
                plot_df["N"].unique()
            )

            ax.set_xticks(xvals)

            # Display powers of two:
            # 2^7, 2^8, ..., rather than 128, 256, ...
            ax.set_xticklabels(
                [
                    rf"$2^{{{int(np.log2(x))}}}$"
                    for x in xvals
                ]
            )

        # ------------------------------------------------------------
        # 6. Y axis
        # ------------------------------------------------------------
        if ylog:
            ax.set_yscale("log")

            # Do NOT set lower limit to zero on a log scale.
            positive_lower = plot_df.loc[
                plot_df["ci_low"] > 0,
                "ci_low",
            ]

            if len(positive_lower) > 0:
                ymin = positive_lower.min() * 0.8
            else:
                ymin = (
                    plot_df.loc[
                        plot_df["mean"] > 0,
                        "mean",
                    ].min()
                    * 0.5
                )

            ymax = (
                plot_df["ci_high"].max()
                * 1.2
            )

            ax.set_ylim(
                bottom=ymin,
                top=ymax,
            )

        else:
            ymax = (
                plot_df["ci_high"].max()
                * 1.05
            )

            ax.set_ylim(
                bottom=0,
                top=ymax,
            )

        # ------------------------------------------------------------
        # 7. Labels / legend / formatting
        # ------------------------------------------------------------
        ax.set_xlabel(
            "Number of samples $N$"
        )

        ax.set_ylabel(
            "Mean absolute error"
        )

        ax.legend(
            title="Shadow model",
            frameon=True,
        )

        # Slightly smaller tick labels to prevent overlap
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
        )

        ax.grid(
            True,
            which="major",
            alpha=0.3,
        )

        # Important for avoiding clipped / overlapping labels
        fig.tight_layout()

        # ------------------------------------------------------------
        # 8. Save
        # ------------------------------------------------------------
        if path_save is not None:
            fig.savefig(
                path_save,
                bbox_inches="tight",
                dpi=300,
            )

        # Uncomment if you want the plot to pop up interactively
        # plt.show()

        return fig, ax



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

def example_experiment(n, k):
    if not 0 < k < n:
        raise ValueError("The tuned binomial experiment requires 0 < k < n")

    q = k/n
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 13)]
    N_obs = 50
    reps = 10
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs,
                                             k_local=k, padding_indices=[0,3]) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQSTEx1", "results/PauliEx", "results/CliffordEx",
                  f"results/SEEQSTEx2_q{k}of{n}"]
    experiment_names = []
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiment_names = ["SEEQST unif", "Pauli", "Clifford", "SEEQST binom"]
    experiments_list=[]
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i],
                                            state_batch_size=1,
                                            experiment_name=experiment_names[i]
                                            )
        experiments_list.append(exp)
    # For observables with exactly k X/Y factors, q=k/n maximizes the
    # probability q**k * (1-q)**(n-k) of drawing the useful block mask.
    experiments_list.append(ShadowScalingExperiment(SEEQSTShadow, HRState, shadow_args={"distribution": partial(binomial, q=q)},
                                                    state_name=state_name,
                                                    N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                                    verbose=True, key=key2, path_save=data_paths[3],
                                                    state_batch_size=1,
                                                    experiment_name=experiment_names[3]
                                                    ))
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)),
                             xlog=True, ylog=True)

def example_experiment_batched(n, k, bs_state=1, bs_obs=1, bs_N=1):
    if not 0 < k < n:
        raise ValueError("The tuned binomial experiment requires 0 < k < n")

    q = k/n
    ns = list(range(n,n+1))
    Ns = [2**k for k in range(7, 13)]
    N_obs = 50
    reps = 10
    key = random.PRNGKey(12345)
    key, key2 = random.split(key)
    obs_lists = [PauliObservable.init_random(key2, sample_indices=[1,2], n=n, N=N_obs,
                                             k_local=k, padding_indices=[0,3]) for n in ns]
    state_name = "HR_state"
    data_paths = ["results/SEEQSTEx1", "results/PauliEx", "results/CliffordEx",
                  f"results/SEEQSTEx2_q{k}of{n}"]
    experiment_names = []
    shadow_classes = [SEEQSTShadow, PauliShadow, CliffordShadow]
    experiment_names = ["SEEQST unif", "Pauli", "Clifford", "SEEQST binom"]
    experiments_list=[]
    for i, shadow_cls in enumerate(shadow_classes):
        key, key2 = random.split(key)
        shadow_args = {}#{"device": "lightning.gpu"} # {"full_setting": False} if shadow_cls == SEEQSTShadow else {}
        exp = ShadowScalingExperiment(shadow_cls, HRState, shadow_args=shadow_args, state_name=state_name,
                                            N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                            verbose=True, key=key2, path_save=data_paths[i],
                                            state_batch_size=bs_state, batch_size_obs=bs_obs, batch_size_N=bs_N,
                                            experiment_name=experiment_names[i]
                                            )
        experiments_list.append(exp)
    # For observables with exactly k X/Y factors, q=k/n maximizes the
    # probability q**k * (1-q)**(n-k) of drawing the useful block mask.
    experiments_list.append(ShadowScalingExperiment(SEEQSTShadow, HRState, 
                                                    shadow_args={"distribution": partial(binomial, q=q)},
                                                    state_name=state_name,
                                                    N_list=Ns, n_list=ns, obs_lists=obs_lists, N_state_reps=reps,
                                                    verbose=True, key=key2, path_save=data_paths[3],
                                                    state_batch_size=bs_state, batch_size_obs=bs_obs, batch_size_N=bs_N,
                                                    experiment_name=experiment_names[3]
                                                    ))
    experiments = Experiments(experiments_list)
    #experiments.plot(x="N", y="Error", hue="Observable", style="Shadow model", markers=['o']*len(shadow_classes))
    experiments.plot_avg_var(path_save="results/scaling_batched_n{}_Nobs{}_reps{}_N{}.pdf".format(n,N_obs,reps,max(Ns)),
                             xlog=True, ylog=True)

if __name__ == "__main__":
    """test_multiple_hom_paulis(4)
    test_multiple_hom_paulis_klocal(4,2)
    test_multiple_hom_paulis(5)
    test_multiple_hom_paulis_klocal(5,2)
    test_multiple_hom_paulis(10)
    test_multiple_hom_paulis_klocal(10,2)"""
    # XY_combos(6)
    example_experiment_batched(15, 5, 1, 50, 4096)
    # example_experiment(15, 14)
