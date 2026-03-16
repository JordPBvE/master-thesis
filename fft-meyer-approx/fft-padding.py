import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

class Distribution:
    def __init__(self, pdf, char):
        self.pdf = pdf
        self.char = char

class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        super().__init__(
            lambda ts : stats.gaussian.pdf(ts, mu, sigma),
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

def meyer_T3s(distr : Distribution, nu, m, N):
    xis = np.arange(N) / N

    H = lambda xi: distr.char((2**(m+1)/3) * np.pi * (xi+1)) * np.cos((np.pi/2) * nu(xi))
    ifft_input = np.concatenate((H(xis), np.zeros(2*N)))

    return ifft_input

def meyer_T2s(distr : Distribution, m, N):
    xis = np.arange(N) / N

    G = lambda xi: distr.char((2**(m+1)/3) * np.pi * xi) 
    ifft_input = np.concatenate((G(xis), np.zeros(2*N)))

    return ifft_input

def meyer_coefficients(distr, nu, m, N):
    T2_xs = meyer_T2s(distr, m, N)
    T3_xs = meyer_T3s(distr, nu, m, N)

    ifft_input = np.vstack((T2_xs, T3_xs)) 
    ifft_output = np.fft.ifft(ifft_input)
    full_ifft_output = np.concatenate((ifft_output[:, 1:], ifft_output), axis=1)

    T2_ifft, T3_ifft = full_ifft_output

    ks = np.arange(-(3*N-1), 3*N)
    factor_2 = 2**(2+m/2) * np.pi
    factor_3 = 2**(1+m/2) * np.pi * np.exp(2j * np.pi * ks / 3)

    T2s = factor_2 * np.real(T2_ifft)
    T3s = factor_3 * (T3_ifft)
    
    cmks = 1 / (2 * np.pi) * (T2s + 2 * np.real(T3s))
    
    return cmks

def phi_hat_matrix(phi_hat_function, m, N, ws):
    ks = np.arange(-(3*N-1), 3*N) 
    pre_term = (1 / 2**(m/2)) * np.exp(-1j * (ws[:, np.newaxis] * ks[np.newaxis, :]) / (2**m))
    
    return pre_term * phi_hat_function(np.abs(ws) / (2**m))[:, np.newaxis]

def phi_hat_meyer(ws, nu):
    abs_omega = np.abs(ws)

    result = np.zeros_like(abs_omega, dtype=float)
    
    cond1 = (abs_omega <= (2 * np.pi / 3))
    result[cond1] = 1.0
    
    cond2 = (abs_omega > (2 * np.pi / 3)) & (abs_omega <= (4 * np.pi / 3))
    abs_omega_subset = abs_omega[cond2]
    
    x_for_nu = (3 / (2 * np.pi)) * abs_omega_subset - 1
    result[cond2] = np.cos((np.pi / 2) * nu(x_for_nu))
    
    return result

def nu_lin(x):
    func = lambda t: t
    return np.piecewise(x, [x <= 0, x >= 1], [0, 1, func])

def nu_poly3(x):
    func = lambda t: 3*t**2 - 2*t**3
    return np.piecewise(x, [x <= 0, x >= 1], [0, 1, func])

def nu_poly5(x):
    func = lambda t: t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
    return np.piecewise(x, [x <= 0, x >= 1], [0, 1, func])

def compute_convergence_errors(distr, nu, m, N, ws):
    cmks = meyer_coefficients(distr, nu, m, N)
    phi_hat_nu = lambda w : phi_hat_meyer(w, nu)
    matrix = phi_hat_matrix(phi_hat_nu, m, N, ws)

    weighted_matrix = matrix * cmks[np.newaxis, :]
    
    analytic_vals = distr.char(ws)
    errors = np.zeros(3 * N)
    
    center_idx = 3 * N - 1
    
    projection = weighted_matrix[:, center_idx].copy()
    errors[0] = np.linalg.norm(analytic_vals - projection, 2)

    for K in range(1, 3 * N):
        projection += weighted_matrix[:, center_idx - K] + weighted_matrix[:, center_idx + K]
        errors[K] = np.linalg.norm(analytic_vals - projection, 2)
        
    return errors

N = 2**12
m = 5

w_max = 2**(m+2) * np.pi / 3 
ws = np.linspace(-w_max, w_max, 1000)
distribution = NIG(alpha=5.0, beta=0.0, mu=0.3, delta=0.3)
# distribution = Gaussian(1, 0.5)

plt.figure(figsize=(10, 6))
for name, nu in [("Linear", nu_lin), ("Poly3", nu_poly3), ("Poly5", nu_poly5)]:
    print(f"Processing {name}...")
    errs = compute_convergence_errors(distribution, nu, m, N, ws)
    plt.semilogy(errs, label=name)

plt.title(f"Meyer Convergence Errors (N={N})")
plt.xlabel("Number of terms (K)")
plt.ylabel("L2 Error")
plt.legend()
plt.grid(True, which="major", ls="-", alpha=0.5)
plt.show()