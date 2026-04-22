import numpy as np
from scipy.integrate import quad

class Distribution:
    def __init__(self, char, name):
        self.char = char
        self.name = name

class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        super().__init__(
            char = lambda ws: np.exp(-1j * mu * ws - 0.5 * (sigma**2) * (ws**2)),
            name = f"N({mu}, {sigma})"
        )

class Cauchy(Distribution):
    def __init__(self, mu, theta):
        super().__init__(
            char = lambda w : np.exp(-1j * mu * w - theta * np.abs(w)),
            name = f"Cauchy({mu}, {theta})"
        )

class NIG(Distribution):
    def __init__(self, alpha, beta, mu, delta):
        gamma = np.sqrt(alpha**2 - beta**2)   
        super().__init__(
            char = lambda ws: np.exp(-mu * 1j * ws + delta * (gamma - np.sqrt(alpha**2 - (beta - 1j * ws)**2))),
            name = f"NIG({alpha}, {beta}, {mu}, {delta})"
        )

class Laplace(Distribution):
    def __init__(self, mu, b):
        super().__init__(
            char = lambda ws: np.exp(-1j * mu * ws) / (1 + (b * ws)**2),
            name = f"Laplace({mu}, {b})"
        )

class VarianceGamma(Distribution):
    def __init__(self, mu, sigma, nu, theta):
        g = lambda ws: 1 + 1j * theta * nu * ws + 0.5 * sigma**2 * nu * ws**2
        super().__init__(
            char = lambda ws: np.exp(-1j * mu * ws) * (g(ws)**(-1/nu)),
            name = f"VG({mu}, {sigma}, {nu}, {theta})"
        )


densities = [
    Cauchy(0.0, 1.0),
    Cauchy(0.0, 0.5),
    Cauchy(0.0, 0.1),
    NIG(5.0, 4.9, 0.0, 0.3),
    NIG(2.0, 0.0, 0.0, 1.0),
    NIG(1.5, -0.5, 0.0, 0.5),
    NIG(1.01, 1.0, 0.0, 1.0),
    Laplace(0.0, 3.0),
    Laplace(0.0, 1.0),
    Laplace(0.0, 0.1),
    VarianceGamma(0.0, 0.2, 0.8, -0.2),
    VarianceGamma(0.0, 0.5, 2.0, 0.0),
    VarianceGamma(0.0, 1.0, 1.0, 0.5),
    VarianceGamma(50.0, 0.5, 1.0, 0.0),
    VarianceGamma(0.0, 0.1, 5.0, 0.0)
]

ms = range(1, 7)

pad_width = max(len(d.name) for d in densities)
for density in densities:
    # 2. The '<' left-aligns the string to the exact pad_width
    print(f"{density.name:<{pad_width}}", end='')
    for m in ms:
        a = 2**(m+1) * np.pi / 3
        b = 2**(m+2) * np.pi / 3

        integrand = lambda w: np.abs(density.char(w))**2

        total, _ = quad(integrand, 0, b, limit=10000, epsabs=1.49e-13, epsrel=1.49e-13)
        tail,  _ = quad(integrand, a, b, limit=10000, epsabs=1.49e-13, epsrel=1.49e-13)

        rho = tail / total
        print(f"| m={m}, rho={rho:.2e} ", end="")
    print("")