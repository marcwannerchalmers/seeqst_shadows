from typing import List, Dict

from jaxed.shadow import Shadow, PauliShadow
from jaxed.tools.observable import Observable, PauliObservable
from tools.estimator import Estimator
from jaxed.tools.state import State, HRState
from GP.online import OnlineAlgo
from GP.model import init_GP_model
from GP.train import train_gp
from GP.data import get_train_set
from numpy.typing import NDArray
from jax import numpy as jnp
import jax
from jax import Array, random
from flax import struct
import gpytorch
import time
import torch
from torch import Tensor
import numpy as np


class GPShadow(Shadow):

    model_states: List[dict] | None = struct.field(pytree_node=False,
                                                    default=None)
    likelihood_states: List[dict] | None = struct.field(pytree_node=False,
                                                                      default=None)
    # Specifications for different args, training, etc.
    cfg: Dict = struct.field(pytree_node=False,
                             default_factory=dict)

    @classmethod
    def init(cls, key: Array, n: int, N: int, N_state_reps: int=1, 
                sample_idx_range: Array=jnp.array([]), 
                estimator: Estimator=Estimator(), optimizer: OnlineAlgo=None, cfg={}, *args, **kwargs):

        if optimizer is None:
            instance = super().init(key, n, N, N_state_reps, jnp.array([0,3]), estimator, *args, **kwargs)
            return instance.replace(cfg=cfg)

        else:
            raise NotImplementedError()

        
    @staticmethod
    def U_fun(ind: Array, **kwargs):
        return PauliShadow.U_fun(ind, **kwargs)

    @staticmethod
    def Udag_fun(ind: Array, **kwargs):
        return PauliShadow.U_fun(ind, **kwargs)

# train the GP here. Maybe rename this function in the base
    def create_snapshots(self) -> Shadow:
        return self

    def train_models(self, Ns: Array):
        model_states = []
        likelihood_states = []
        qstate = 0
        for N in Ns:
            # TODO: Turn into 'get_train_loader'
            train_loader = get_train_set(self.indices[qstate,:N],
                                        self.outcomes[qstate,:N])
            model, train_x, train_y, likelihood = init_GP_model(train_loader, 
                                                                self.cfg["kernel_type"],
                                                                self.cfg["kernel_args"])


            model, likelihood = train_gp(self.cfg,
                                    model, 
                                    likelihood, 
                                    train_x,
                                    train_y)              
            
            model_state = model.state_dict()
            likelihood_state = likelihood.state_dict()
            model_states.append(model_state)
            likelihood_states.append(likelihood_state)
        
        return model_states, likelihood_states

    # TODO: Make dependent of N
    def estimate_properties(self, obs: Observable, Ns: Array = jnp.array([0]), 
                            batch_size: int | None = None) -> Array:
        model_states, likelihood_states = self.train_models(Ns)
        x = Tensor(obs.params)
        outs = []
        for qstate in range(self.N_state_reps):
            outs_inner = []
            for i, N in enumerate(Ns):
                # TODO: Turn into 'get_train_loader'
                train_loader = get_train_set(self.indices[qstate,:N],
                                            self.outcomes[qstate,:N])
                model, train_x, train_y, likelihood = init_GP_model(train_loader, 
                                                                    self.cfg["kernel_type"],
                                                                    self.cfg["kernel_args"])

                
                model.load_state_dict(model_states[i])
                likelihood.load_state_dict(likelihood_states[i])

                model.eval()
                likelihood.eval()

                outs_inner.append(likelihood(model(x)).mean)
            outs.append(torch.stack(outs_inner))
        out = torch.stack(outs).permute(0,2,1)
        return jnp.array(out.detach().numpy())

    @classmethod
    def _inverse_channel(cls, n: int, inv_oc: Array, obs: Observable) -> Array:
        raise NotImplementedError()

    @classmethod
    def _inverse_circuit(cls, n: int, ind_axis, obs_treedef, U_treedef):
        raise NotImplementedError()

    @classmethod
    def inverse_circuit_clifford(cls, outcome: Array, ind: Array):
        raise NotImplementedError()
        

def testGPshadow():
    n = 5
    paulis = ["X"*n,"Y"*n, "Z"*n]
    N = 2**14
    N_state_reps = 5
    obs = PauliObservable.init(paulis)
    key = jax.random.PRNGKey(1234)
    states = HRState.init_random(key, N_state_reps, n)
    print("started timing")
    start = time.time()
    shadow = GPShadow.init(key, n, N, N_state_reps, cfg={"kernel_type": "Hamming", "kernel_args": {},
                                                         "share_parameters": True})
    print(shadow.indices.shape)
    shadow = shadow.sample(states)
    end = time.time()
    print(end-start)
    print("sampled")
    start = time.time()
    shadow = shadow.create_snapshots()
    # shadow = shadow.train_models()
    props = shadow.estimate_properties(obs, Ns=[N])
    print(props)
    end = time.time()
    print(states.state_dm.shape)
    print(shadow.ground_truth(states, obs))
    print(end-start)

if __name__ == "__main__":
    testGPshadow()