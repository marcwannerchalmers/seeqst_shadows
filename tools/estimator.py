from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

class Estimator:
    def __init__(self) -> None:
        pass

    def __call__(self, x: NDArray):
        return np.mean(x, axis=-1)

    def createZtype(self) -> ZTypeEstimator:
        return ZTypeEstimator()

# Estimator specifically for Z-type observables
class ZTypeEstimator(Estimator):
    def __init__(self) -> None:
        super().__init__()
        self.weights = np.array([])

    def set_setup(self, n: int, setting_indices: NDArray):
        self.weights = np.where(setting_indices > 0, np.zeros_like(setting_indices), 
                                np.ones_like(setting_indices))
        self.weights = self.weights/(np.sum(self.weights)*2**n)

    def __call__(self, x: NDArray):
        assert len(self.weights) > 0, "Call set_setting_indices before calling the estimator!"
        return np.inner(x, self.weights)
    
class MedianOfMeans(Estimator):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def __call__(self, x: NDArray):
        buckets = np.array_split(x, self.k, axis=-1)
        buckets = np.stack([super().__call__(bucket) for bucket in buckets], axis=-1)
        return np.median(buckets, axis=-1)

    def createZtype(self) -> ZTypeMedianOfMeans:
        return ZTypeMedianOfMeans()

# TODO: Implement Z-type version of this estimator too
class ZTypeMedianOfMeans(ZTypeEstimator):
    pass