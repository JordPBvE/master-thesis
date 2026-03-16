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
    def __init__(self, func):
        self.function          = lambda x : np.piecewise(x, [x < 0, x > 1], [0, 1, func])

nu_lin = Nu(
    lambda x : x, 
)

nu_poly3 = Nu(
    lambda x : 3*x**2 - 2*x**3, 
)

nu_poly5 = Nu(
    lambda x: x**4 * (35 - 84*x + 70*x**2 - 20*x**3),
)

def get_T3_input(distr, nu, m, xis):
    H = lambda xi: distr.char((2**(m+1)/3) * np.pi * (xi+1)) * np.cos((np.pi/2) * nu.function(xi))

    p0 = distr.char(2**(m+1) * np.pi / 3)
    s = lambda xi : (1 - xi) * p0 

    ifft_input = H(xis) - s(xis)
    ifft_input[xis.size//3:] = 0
    return ifft_input

def get_T2_input(distr, m, xis):
    G = lambda xi: distr.char((2**(m+1)/3) * np.pi * xi) 

    p0 = distr.char(0)
    p1 = distr.char(2**(m+1) * np.pi / 3)
    s = lambda xi : p0 + (p1-p0) * xi 

    ifft_input = G(xis) - s(xis)
    ifft_input[xis.size//3:] = 0
    return ifft_input

m = 5
distr = NIG(alpha=5.0, beta=0.0, mu=0.3, delta=0.3)
nu_obj = nu_poly5

xis = np.linspace(0, 3, 600)

czt_input_T2 = get_T2_input(distr, m, xis)
czt_input_T3 = get_T3_input(distr, nu_obj, m, xis)

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Plot T3 Input (H part) 
ax1.plot(xis, np.real(czt_input_T3), '.', markersize=2, color='tab:blue', label='Real')
ax1.plot(xis, np.imag(czt_input_T3), '.', markersize=2, color='tab:cyan', label='Imaginary')
ax1.set_ylabel(r'$H_{\hat f, \nu}(\xi) - s_{H_{\hat f, \nu}}$')
ax1.set_title('CZT input with H')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot T2 Input (G part)
ax2.plot(xis, np.real(czt_input_T2), '.', markersize=2, color='tab:orange', label='Real')
ax2.plot(xis, np.imag(czt_input_T2), '.', markersize=2, color='tab:red', label='Imaginary')
ax2.set_xlabel(r'$\xi$')
ax2.set_ylabel(r'$G_{\hat f}(\xi) - s_{G_{\hat f}}$')
ax2.set_title('CZT input with G')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()