# Manifold Sampling

Sampling algorithms for probability distributions on manifolds using Riemannian geometry. Generates samples from arbitrary curves and surfaces by performing pullback on the input density.

$$
 \int_{\phi(U)} fdV = \int_U  f(\phi(x)) \det(\phi'(x)^T \phi'(x))^{1/2}dx_1\dots dx_k
$$

Sampling on $\mathbb{R}^k$ is done using rejection sampling with a uniform proposal distribution.
This is done under the assumption that pulled-back density is bounded.

<p align="center"><img src="output.png" alt="Torus Sampling" width="500"></p>

## Compilation

To compile python bindings run `make python`. To compile C++ demos run `make ellipse` or `make torus`.

To install the R package:

```bash
cd src/R
R CMD INSTALL .
```

## Usage

See [demo.ipynb](demo.ipynb) for detailed examples and applications
to Monte Carlo integration and estimating the mean Hausdorff distance
between compact manifolds.

### Python

```python
from saman import Manifold
import numpy as np

class Ellipse(Manifold):
    def __init__(self):
        super().__init__(2, 1, 2.5, np.array([0.0]), np.array([2.0 * np.pi]))

    def coord(self, t):
        return np.array([2*np.cos(t[0]), np.sin(t[0])])

    def pushforward(self, t):
        return np.array([[-2*np.sin(t[0])], [np.cos(t[0])]])

ellipse = Ellipse()
samples = ellipse.sample(n_samples=100)
```

### R

```r
library(saman)

coord = function(t) c(2 * cos(t[1]), sin(t[1]))
pushforward = function(t) matrix(c(-2 * sin(t[1]), cos(t[1])), 2, 1)

ellipse = saman.manifold(2, 1, 2.5, c(0), c(2*pi), coord, pushforward)
samples = ellipse$sample(100)
```

### C++

If you want to work with C++ directly see the [examples/](src/example/) folder.

## Todo

- [x] Move core logic to C++
- [x] Create python bindings
- [x] Create R bindings
- [x] Demo library by computing mean Hausdorff distance.
- [ ] Write expository article about how this is done.
