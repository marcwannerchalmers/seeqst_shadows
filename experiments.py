from tools.observable import PauliObservable
from tools.state import HRState
import pennylane as qp
from shadow import SEEQSTShadow
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# TODO: make such that observables are tied to number of qubits
class ShadowScalingExperiment:
    def __init__(self, N_list, n_list, obs_lists:List[List[PauliObservable]]=[]) -> None:
        self.Ns = N_list
        self.ns = n_list
        self.observables_list = []
        if len(obs_lists) == len(self.ns):
            self.observables_list = obs_lists
        self.preds = np.zeros((len(self.Ns), len(self.ns), len(self.observables_list[0])))
        self.gt = np.zeros_like(self.preds)

    def step(self, N, n, observables):
        shadow = SEEQSTShadow(HRState(n), [0, 2**n], full_setting=True)
        shadow.create(N)
        return shadow.predict(observables), shadow.ground_truth(observables)
    
    def run(self):
        for i, N in enumerate(self.Ns):
            for j, (n, observables) in enumerate(zip(self.ns, self.observables_list)):
                self.preds[i,j], self.gt[i,j] = self.step(N, n, observables)
                print(f"N: {N}, n: {n}, first observable: {observables[0].get_name()}, GT: {self.gt[i,j]}, Pred: {self.preds[i,j]}")

    def plot(self, mode="data_scaling", path_save=None):
        if mode == "data_scaling":
            for i, obs in enumerate(self.observables_list[0]):
                for j, n in enumerate(self.ns):
                    plt.plot(np.array(self.Ns), np.abs(self.preds[:,j,i]-self.gt[:,j,i]), 
                            label=f"obs: {obs.get_obs_string()}, n: {n}")
        else:
            for i, obs in enumerate(self.observables_list[0]):
                for j, N in enumerate(self.Ns):
                    plt.plot(np.array(self.ns), np.abs(self.preds[j,:,i]-self.gt[j,:,i]), 
                            label=f"obs: {obs.get_obs_string()}, N: {N}")
        
        plt.legend()
        plt.show()

        if path_save is not None:
            plt.savefig(path_save)

def test_homogenous_paulis():
    ns = list(range(3,4))
    Ns = [2**k for k in range(4, 5)]
    # obs_list = [[PauliObservable(O*n) for O in ["X","Y","Z"]] for n in ns]
    obs_list = [[PauliObservable(O*n) for O in ["X"]] for n in ns]
    experiment = ShadowScalingExperiment(Ns, ns, obs_list)
    experiment.run()
    # experiment.plot()

def main():
    test_homogenous_paulis()


if __name__ == "__main__":
    main()