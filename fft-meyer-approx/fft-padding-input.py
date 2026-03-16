import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class Distribution:
    def __init__(self, pdf, char):
        self.pdf = pdf
        self.char = char

class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        super().__init__(
            lambda ts : stats.norm.pdf(ts, mu, sigma),
            lambda ws : np.exp(- mu * 1j * ws - sigma * np.power(ws, 2))
        )

class Cauchy(Distribution):
    def __init__(self, mu, theta):
        super().__init__(
            lambda ts : stats.cauchy.pdf(ts, mu, theta),
            lambda ws : np.exp(- mu * 1j * ws - theta * np.abs(ws)) 
        )

class NIG(Distribution):
    def __init__(self, alpha, beta, mu, delta):
        gamma = np.sqrt(alpha**2 - beta**2)   
        super().__init__(
            lambda ts: stats.norminvgauss.pdf(ts, a=alpha, b=beta, loc=mu, scale=delta),
            lambda ws: np.exp(-mu * 1j * ws + delta * (gamma - np.sqrt(alpha**2 - (beta - 1j * ws)**2)))
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

def get_T2_input(distr, m, xis):
    fraction = 2**(m+1) * np.pi / 3
    G = lambda xi: distr.char(fraction * xi)

    return G(xis)

def get_T3_input(distr, nu, m, xis):
    fraction = 2**(m+1) * np.pi / 3
    H = lambda xi: distr.char(fraction * (xi + 1)) * np.cos((np.pi/2) * nu.function(xi))
    return H(xis)

# 1. Setup
m = 6
distr = NIG(alpha=5.0, beta=0.0, mu=0.3, delta=0.3)
nu_obj = nu_poly5

# 2. Domain (0 to 3 for xi)
xis = np.linspace(0, 3, 600)

czt_input_T2 = get_T2_input(distr, m, xis)
czt_input_T3 = get_T3_input(distr, nu_obj, m, xis)

# This hard truncation at index 100 (xi = 1.0) creates the discontinuity
czt_input_T2[200:] = 0
czt_input_T3[200:] = 0

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Plot T3 Input (H part) - added markersize=2
ax1.plot(xis, np.real(czt_input_T3), '.', markersize=2, color='tab:blue', label='Real')
ax1.plot(xis, np.imag(czt_input_T3), '.', markersize=2, color='tab:cyan', label='Imaginary')
ax1.set_ylabel(r'$H(\xi)$')
ax1.set_title('FFT input with H')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot T2 Input (G part) - added markersize=2
ax2.plot(xis, np.real(czt_input_T2), '.', markersize=2, color='tab:orange', label='Real')
ax2.plot(xis, np.imag(czt_input_T2), '.', markersize=2, color='tab:red', label='Imaginary')
ax2.set_xlabel(r'$\xi$')
ax2.set_ylabel(r'$G(\xi)$')
ax2.set_title('FFT input with G')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()