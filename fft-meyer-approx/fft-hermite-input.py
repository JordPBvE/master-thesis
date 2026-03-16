import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class Distribution:
    def __init__(self, pdf, char, char_deriv, char_sec_deriv):
        self.pdf = pdf
        self.char = char
        self.char_derivative = char_deriv
        self.char_second_derivative = char_sec_deriv

class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        phi = lambda ws: np.exp(-1j * mu * ws - 0.5 * (sigma**2) * (ws**2))
        super().__init__(
            pdf = lambda ts: stats.norm.pdf(ts, mu, sigma),
            char = phi,
            char_deriv = lambda ws: ((-1j * mu - (sigma**2) * ws) * phi(ws)),
            char_sec_deriv = lambda ws: (
                (-(sigma**2) + (-1j * mu - (sigma**2) * ws)**2) * phi(ws)
            )
        )

class NIG(Distribution):
    def __init__(self, alpha, beta, mu, delta):
        gamma = np.sqrt(alpha**2 - beta**2)   

        phi = lambda ws: np.exp(-mu * 1j * ws + delta * (gamma - np.sqrt(alpha**2 - (beta - 1j * ws)**2)))
        S = lambda ws: np.sqrt(alpha**2 - (beta - 1j * ws)**2)
        f_prime = lambda ws: -1j * mu - (1j * delta * (beta - 1j * ws)) / S(ws)
        f_sec_prime = lambda ws: -(delta * alpha**2) / (S(ws)**3)

        super().__init__(
            pdf = lambda ts: stats.norminvgauss.pdf(ts, a=alpha, b=beta, loc=mu, scale=delta),
            char = phi,
            char_deriv = lambda ws: f_prime(ws) * phi(ws),
            char_sec_deriv = lambda ws: (f_sec_prime(ws) + f_prime(ws)**2) * phi(ws)
        )

class Nu:
    def __init__(self, func , deriv, sec_deriv):
        self.function          = lambda x : np.piecewise(x, [x < 0, x > 1], [0, 1, func])
        self.derivative        = lambda x : np.piecewise(x, [x < 0, x > 1], [0, 0, deriv])
        self.second_derivative = lambda x : np.piecewise(x, [x < 0, x > 1], [0, 0, sec_deriv])

nu_lin = Nu(
    lambda x : x, 
    lambda _ : 1,
    lambda _ : 0 
)

nu_poly3 = Nu(
    lambda x : 3*x**2 - 2*x**3, 
    lambda x : 6*x - 6*x**2,
    lambda x : 6 - 12*x
)

nu_poly5 = Nu(
    lambda x: x**4 * (35 - 84*x + 70*x**2 - 20*x**3),
    lambda x: 140*x**3 - 420*x**4 + 420*x**5 - 140*x**6,
    lambda x: 420*x**2 - 1680*x**3 + 2100*x**4 - 840*x**5
)

def hermite_spline_polynomial(f0, f1, v0, v1, a0, a1):
    c0 = f0
    c1 = v0
    c2 = 0.5 * a0
    c3 = 10 * (f1 - f0) - (6 * v0 + 4 * v1) - 0.5 * (3 * a0 - a1)
    c4 = -15 * (f1 - f0) + (8 * v0 + 7 * v1) + 0.5 * (3 * a0 - 2 * a1)
    c5 = 6 * (f1 - f0) - 3 * (v0 + v1) - 0.5 * (a0 - a1)
    return np.array([c0, c1, c2, c3, c4, c5])

def get_T2_input(distr, m, xis):
    fraction = 2**(m+1) * np.pi / 3
    G = lambda xi: distr.char(fraction * xi)

    # Boundaries
    p0 = distr.char(0)
    p1 = distr.char(fraction)
    v0 = fraction * distr.char_derivative(0)
    v1 = fraction * distr.char_derivative(fraction)
    a0 = fraction**2 * distr.char_second_derivative(0)
    a1 = fraction**2 * distr.char_second_derivative(fraction)

    # Spline
    coeffs = hermite_spline_polynomial(p0, p1, v0, v1, a0, a1)
    powers = np.arange(6)
    s = lambda xi: np.sum(coeffs * np.power(xi[:, None], powers), axis=-1)

    ifft_input = G(xis) - s(xis)
    ifft_input[xis.size//3:] = 0

    return ifft_input

def get_T3_input(distr, nu, m, xis):
    fraction = 2**(m+1) * np.pi / 3
    
    H = lambda xi: distr.char(fraction * (xi + 1)) * np.cos((np.pi/2) * nu.function(xi))

    # Boundaries (Copied exactly from meyer_T3s)
    p0 = distr.char(fraction)
    p1 = 0
    v0 = fraction * distr.char_derivative(fraction)
    v1 = -distr.char(2 * fraction) * (np.pi / 2) * nu.derivative(1)
    
    a0 = fraction**2 * distr.char_second_derivative(fraction) - \
         (np.pi**2 / 4) * distr.char(fraction) * (nu.derivative(0))**2
         
    a1 = - np.pi * fraction * distr.char_derivative(2*fraction) * nu.derivative(1) - \
         (np.pi/2) * distr.char(2*fraction) * nu.second_derivative(1)

    # Spline
    coeffs = hermite_spline_polynomial(p0, p1, v0, v1, a0, a1)
    powers = np.arange(6)
    s = lambda xi: np.sum(coeffs * np.power(xi[:, None], powers), axis=-1)

    ifft_input = H(xis) - s(xis)
    ifft_input[xis.size//3:] = 0

    return ifft_input

# --- Visualization Script ---

# 1. Setup
m = 6
distr = NIG(alpha=5.0, beta=0.0, mu=0.3, delta=0.3)
nu_obj = nu_poly5

xis = np.linspace(0, 3, 600)

czt_input_T2 = get_T2_input(distr, m, xis)
czt_input_T3 = get_T3_input(distr, nu_obj, m, xis)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Plot T3 Input (H part) - added markersize=2
ax1.plot(xis, np.real(czt_input_T3), '.', markersize=2, color='tab:blue', label='Real')
ax1.plot(xis, np.imag(czt_input_T3), '.', markersize=2, color='tab:cyan', label='Imaginary')
ax1.set_ylabel(r'$H_{\hat f, \nu}(\xi) - s^5_{H_{\hat f, \nu}}$')
ax1.set_title('IFFT input with H')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot T2 Input (G part) - added markersize=2
ax2.plot(xis, np.real(czt_input_T2), '.', markersize=2, color='tab:orange', label='Real')
ax2.plot(xis, np.imag(czt_input_T2), '.', markersize=2, color='tab:red', label='Imaginary')
ax2.set_xlabel(r'$\xi$')
ax2.set_ylabel(r'$G_{\hat f}(\xi) - s^5_{G_{\hat f}}$')
ax2.set_title('IFFT input with G')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()