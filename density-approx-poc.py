import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfi, erf
from scipy import integrate
import scipy.stats as stats

def phi_shannon(ts):
    return np.sinc(ts)

def phi_hat_shannon(ws):
    return np.where(abs(ws/(2*np.pi))<=0.5, 1, 0)

def phi_meyer(ts):
    numerator = np.sin(2 * np.pi * ts / 3) + (4 / 3) * ts * np.cos(4 * np.pi * ts / 3)
    denominator = np.pi * ts - (16 * np.pi / 9) * ts**3
    
    result = np.where(
        np.isclose(ts, 0),
        (2/3 + 4 / (3 * np.pi)),
        numerator / denominator
    )
    return result

def phi_hat_meyer(ws):
    abs_omega = np.abs(ws)
    
    result = np.zeros_like(abs_omega, dtype=float)
    
    cond1 = (abs_omega <= (2 * np.pi / 3))
    result[cond1] = 1.0
    
    cond2 = (abs_omega > (2 * np.pi / 3)) & (abs_omega <= (4 * np.pi / 3))
    abs_omega_subset = abs_omega[cond2]

    def nu(x):
        result = np.copy(x)
        result[x <= 0] = 0.0
        result[x >= 1] = 1.0
        return result
    
    x_for_nu = (3 / (2 * np.pi)) * abs_omega_subset - 1
    result[cond2] = np.cos((np.pi / 2) * nu(x_for_nu))
    
    return result

def G(a, b, x):
    term1 = 0.5
    term2 = np.sqrt(np.pi / a)
    term3 = np.exp(-b**2 / (4 * a))
    term4_input = np.sqrt(a) * x + 0.5 * b / np.sqrt(a)
    term4 = erfi(term4_input)
    return term1 * term2 * term3 * term4

def gauss_shannon_coefficients(sigma, mu, m, k_max):
    ks = np.arange(-k_max, k_max + 1) 
    factor = 1 / (np.pi * 2**(1 + m / 2))

    order_2_term = -0.5 * (sigma**2) + 0j
    order_0_term = -1j * (mu - (ks / 2**m))

    func = lambda x: G(order_2_term, order_0_term, x)
    out = factor * (func(2**m * np.pi) - func(-2**m * np.pi))
    
    return out

def gauss_meyer_coefficients(sigma, mu, m, k_max):
    ks = np.arange(-k_max, k_max + 1) 
    factor = 1j / 2**(1 + m / 2)

    order_2_term = -0.5 * (sigma**2) + 0j

    order_1_term_p = -1j * (mu - (ks / 2**m) + (3 / 2**(m + 2)))
    order_1_term_m = -1j * (mu - (ks / 2**m) - (3 / 2**(m + 2)))
    order_0_term = -1j * (mu - (ks / 2**m))

    T1_f = lambda x: G(order_2_term, order_1_term_m, x) - G(order_2_term, order_1_term_p, x)
    T2_f = lambda x: G(order_2_term, order_0_term, x)
    T3_f = lambda x: G(order_2_term, order_1_term_p, x) - G(order_2_term, order_1_term_m, x)

    T1 = factor * (T1_f(-2**(m + 1) / 3) - T1_f(-2**(m + 2) / 3))
    T2 = (1 / 2**(m / 2)) * (T2_f(2**(m + 1) / 3) - T2_f(-2**(m + 1) / 3))
    T3 = factor * (T3_f(2**(m + 2) / 3) - T3_f(2**(m + 1) / 3))

    cmks = (1 / (2 * np.pi)) * (T1 + T2 + T3)
    return cmks

def project(cmks, scalingfunction, m, K, ts):
    ks = np.arange(-K, K+ 1)

    args = (2**m * ts[:, np.newaxis]) - ks[np.newaxis, :]
    phi_matrix = 2**(m/2) * scalingfunction(args) 
    
    projection = np.dot(phi_matrix, cmks)

    return np.real(projection)

def project_fourier(cmks, phi_hat_function, m, K, ws):
    ks = np.arange(-K, K + 1) # Shape: (num_ks,)

    phi_hat_scaled = (1 / 2**(m/2)) * phi_hat_function(ws / (2**m))
    exponent_matrix = -1j * (ws[:, np.newaxis] * ks[np.newaxis, :]) / (2**m)
    exp_matrix = np.exp(exponent_matrix)
    
    sum_term = np.dot(exp_matrix, cmks)
    fourier_approx = phi_hat_scaled * sum_term
    
    return fourier_approx

K = 5
m = 2
ts = np.linspace(-5, 5, 300)
ws = np.linspace(-5, 5, 300)

mu = 0
sigma = 1

analytic_pdf = stats.norm.pdf(ts, mu, sigma)
analytic_char = np.exp(-1j*mu*ws -0.5*sigma**2 * np.square(ws))

# shannon
cmks_shannon = gauss_shannon_coefficients(sigma, mu, m, K)
projection_shannon =  project(cmks_shannon, phi_shannon, m, K, ts)
fourier_projection_shannon = project_fourier(cmks_shannon, phi_hat_shannon, m, K, ws)

# meyer
cmks_meyer = gauss_meyer_coefficients(sigma, mu, m, K)
projection_meyer =  project(cmks_meyer, phi_meyer, m, K, ts)
fourier_projection_meyer = project_fourier(cmks_meyer, phi_hat_meyer, m, K, ws)

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# Top-left plot: Shannon Density Approximation
ax[0, 0].plot(ts, projection_shannon, 'k', linewidth=2, label='Shannon approximation')
ax[0, 0].plot(ts, analytic_pdf, 'r-.', linewidth=2, label = 'Analytic Gaussian')
ax[0, 0].set_title("Shannon Density Approximation (Time)")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("$P_mf$")
ax[0, 0].grid(True)
ax[0, 0].legend()

# Top-right plot: Meyer Density Approximation
ax[0, 1].plot(ts, projection_meyer, 'b', linewidth=2, label='Meyer approximation')
ax[0, 1].plot(ts, analytic_pdf, 'r-.', linewidth=2, label = 'Analytic Gaussian')
ax[0, 1].set_title("Meyer Density Approximation (Time)")
ax[0, 1].set_xlabel("x")
ax[0, 1].set_ylabel("$P_mf$")
ax[0, 1].grid(True)
ax[0, 1].legend()

# Bottom-left plot: Shannon Fourier Approximation
ax[1, 0].plot(ws, fourier_projection_shannon, 'k', linewidth=2, label='Shannon approx. (Fourier)')
ax[1, 0].plot(ws, np.real(analytic_char), 'r-.', linewidth=2, label = 'Analytic Char. Func. (Real)')
ax[1, 0].set_title("Shannon Fourier Approximation")
ax[1, 0].set_xlabel("$\omega$")
ax[1, 0].set_ylabel("$\hat{P}_mf(\omega)$")
ax[1, 0].grid(True)
ax[1, 0].legend()

# Bottom-right plot: Meyer Fourier Approximation
ax[1, 1].plot(ws, fourier_projection_meyer, 'b', linewidth=2, label='Meyer approx. (Fourier)')
ax[1, 1].plot(ws, np.real(analytic_char), 'r-.', linewidth=2, label = 'Analytic Char. Func. (Real)')
ax[1, 1].set_title("Meyer Fourier Approximation")
ax[1, 1].set_xlabel("$\omega$")
ax[1, 1].set_ylabel("$\hat{P}_mf(\omega)$")
ax[1, 1].grid(True)
ax[1, 1].legend()

plt.show()