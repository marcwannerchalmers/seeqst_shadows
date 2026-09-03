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
                for k in observables.params.shape[0]:
                    for i, N in enumerate(experiment.Ns):
                        data["Shadow model"].append(re.search(r"\.(.*?)\'", str(experiment.shadow_cls))[1])
                        data["Observable"].append(observables.obs_string(observables.params[k]))
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