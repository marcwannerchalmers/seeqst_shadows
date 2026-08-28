from typing import List

from shadow import Shadow
from tools.estimator import Estimator, GPEstimator
from tools.state import State
from GP.online import OnlineAlgo
from numpy.typing import NDArray

class GPShadow(Shadow):
    def __init__(self, state: State, gate_indices: List, optimizer: OnlineAlgo=None, **GP_params) -> None:
        estimator = GPEstimator(**GP_params)
        super().__init__(state, gate_indices, estimator)

    # either offline or online, depending on whether algo is None
    def init(self, N: int):
        pass

    def predict(self, obs_list)->NDArray:
        pass

    