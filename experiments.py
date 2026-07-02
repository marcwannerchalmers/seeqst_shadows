from tools.observable import PauliObservable
from tools.state import HRState
import pennylane as qp
from shadow import SEEQSTShadow
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# TODO: make such that observables are tied to number of qubits
class ShadowScalingExperiment:
    def __init__(self, shadow_cls, state_cls, shadow_args={}, state_args = {},
                  N_list=[], n_list=[], obs_lists:List[List[PauliObservable]]=[],
                  gate_indices=lambda n: [0,2**n]) -> None:
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

    def step(self, N, n, observables):
        # shadow = SEEQSTShadow(HRState(n), [0, 2**n], full_setting=False)
        shadow = self.shadow_cls(self.state_cls(n, **self.state_args), 
                                 self.gate_indices(n), **self.shadow_args)
        shadow.create(N)
        return shadow.predict(observables), shadow.ground_truth(observables)
    
    def run(self):
        for i, N in enumerate(self.Ns):
            for j, (n, observables) in enumerate(zip(self.ns, self.observables_list)):
                self.preds[i,j], self.gt[i,j] = self.step(N, n, observables)
                print(f"N: {N}, n: {n}, first observable: {observables[0].get_name()}, GT: {self.gt[i,j]}, Pred: {self.preds[i,j]}")

    def save_results(self, path: str):
        res = np.stack([self.gt, self.preds])
        np.savetxt(path, res)

    def load_results(self, path: str):
        self.gt, self.preds = np.loadtxt(path)

    def plot(self, mode="data_scaling", path_save=None):
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
        
        plt.legend()
        plt.show()

        if path_save is not None:
            plt.savefig(path_save)

def test_homogenous_paulis():
    ns = list(range(4,5))
    Ns = [2**k for k in range(12, 13)]
    # obs_list = [[PauliObservable(O*n) for O in ["X","Y","Z"]] for n in ns]
    obs_list = [[PauliObservable(O*n) for O in ["X"]] for n in ns]
    experiment = ShadowScalingExperiment(SEEQSTShadow, HRState, {"full_setting":False}, 
                                         Ns, ns, obs_list)
    experiment.run()
    # experiment.plot()

def main():
    test_homogenous_paulis()


if __name__ == "__main__":
    main()