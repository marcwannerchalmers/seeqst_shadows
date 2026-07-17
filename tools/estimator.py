from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

class Estimator:
    def __init__(self) -> None:
        pass

    def __call__(self, x: NDArray):
        return np.mean(x, axis=-1)
    
class MedianOfMeans(Estimator):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def __call__(self, x: NDArray):
        buckets = np.array_split(x, self.k, axis=-1)
        buckets = np.stack([super().__call__(bucket) for bucket in buckets], axis=-1)
        return np.median(buckets, axis=-1)