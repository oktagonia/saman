from typing import Any, Callable
from numpy.typing import NDArray
from math import prod
import numpy as np

Real = np.floating
Coord = Callable[[NDArray[Real]], NDArray[Real]]
Pushforward = Callable[[NDArray[Real]], NDArray[Real]] 
PDF = Callable[[NDArray[Real]], Real]
Bounds = list[tuple[Real, Real]]

class Manifold:
    def __init__(self, coords: Coord, pushforward: Pushforward, dim: int, bounds: Bounds):
        self.coords = coords
        self.pushforward = pushforward
        self.dim = dim
        self.bounds = bounds
        self.param_vol = prod(b[1] - b[0] for b in bounds)

    def metric(self, p: NDArray[Real]):
        jacobian = self.pushforward(p)
        return jacobian.T @ jacobian

    def volume_form(self, p: NDArray[Real]):
        return np.linalg.det(np.sqrt(self.metric(p)))

    def _param_samples(self, n_samples: int, M: Real):
        samples = []
        lows, highs = zip(*self.bounds)

        while len(samples) < n_samples:
            x = np.random.uniform(lows, highs, size=(self.dim,))
            u = np.random.uniform()
            if u * M <= self.volume_form(x):
                samples.append(x)

        return samples

    def sample(self, n_samples: int, M: Real):
        param_samples = self._param_samples(n_samples, M)
        return [self.coords(x) for x in param_samples]
