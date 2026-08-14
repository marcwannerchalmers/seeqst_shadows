from qiskit._accelerate.synthesis.clifford import random_clifford_tableau
from numpy.random import default_rng
import numpy as np

if __name__ == "__main__":
    n = 5
    rng = default_rng()
    seed = rng.integers(100000, size=1, dtype=np.uint64)[0]
    tableau = random_clifford_tableau(n, seed=seed)
    print(tableau)