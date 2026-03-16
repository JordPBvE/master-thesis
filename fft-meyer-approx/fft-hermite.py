import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.signal import czt
from scipy.special import factorial

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

def I_QH(k, coeffs, a): 
    k_safe = np.where(k == 0, 1.0, k)
    ns = np.arange(6) 

    n_idx = ns[:, np.newaxis]
    m_idx = ns[np.newaxis, :]
    with np.errstate(invalid='ignore'):
        full_calc = factorial(m_idx) / factorial(m_idx - n_idx)
    weights = np.where(m_idx >= n_idx, full_calc, 0.0)

    term1 = weights @ coeffs
    term0 = factorial(ns) * coeffs

    k_column = k_safe[:, np.newaxis] 
    
    # Now uses the provided 'a' mapping
    numerator = (np.exp(a * k_column) * term1) - term0 
    denominator = np.power(a * k_column, ns + 1)
    sign = np.power(-1, ns)

    safe_result = np.sum(sign * (numerator / denominator), axis=1)

    k_zero_integral = np.sum((1 / np.arange(1, 7)) * coeffs)
    
    return np.where(k == 0, k_zero_integral, safe_result)

def CZT3(xs, N):
    return czt(xs, N, np.exp(2 * np.pi * 1j / (3*N)))

def meyer_T2s(distr : Distribution, m, N):
    xis = np.arange(N) / N

    G = lambda xi: distr.char((2**(m+1)/3) * np.pi * xi) 

    fraction = 2**(m+1) * np.pi / 3

    p0 = distr.char(0)
    p1 = distr.char(2**(m+1) * np.pi / 3)
    v0 = fraction * distr.char_derivative(0)
    v1 = fraction * distr.char_derivative(fraction)
    a0 = fraction**2 * distr.char_second_derivative(0)
    a1 = fraction**2 * distr.char_second_derivative(fraction)

    coeffs = hermite_spline_polynomial(p0, p1, v0, v1, a0, a1)

    powers = np.arange(6)
    s = lambda xi: np.sum(coeffs * np.power(xi[:, None], powers), axis=-1)

    ifft_input = np.concatenate((G(xis) - s(xis), np.zeros(2*N)))

    return ifft_input, coeffs

def meyer_T3s(distr : Distribution, nu : Nu, m, N):
    xis = np.arange(N) / N

    H = lambda xi: distr.char((2**(m+1)/3) * np.pi * (xi+1)) * np.cos((np.pi/2) * nu.function(xi))

    fraction = 2**(m+1) * np.pi / 3
    p0 = distr.char(fraction)
    p1 = 0
    v0 = fraction * distr.char_derivative(fraction)
    v1 = -distr.char(2 * fraction) * (np.pi / 2) * nu.derivative(1)
    a0 = fraction**2 * distr.char_second_derivative(fraction) - (np.pi**2 / 4) * distr.char(fraction) * (nu.derivative(0))**2
    a1 = - np.pi * fraction * distr.char_derivative(2*fraction) * nu.derivative(1) - (np.pi/2) * distr.char(2*fraction) * nu.second_derivative(1)

    coeffs = hermite_spline_polynomial(p0, p1, v0, v1, a0, a1)

    powers = np.arange(6)
    s = lambda xi: np.sum(coeffs * np.power(xi[:, None], powers), axis=-1)

    ifft_input = np.concatenate((H(xis) - s(xis), np.zeros(2*N)))

    return ifft_input, coeffs

def shannon_Ts(distr : Distribution, m, N):
    xis = np.arange(N) / N

    # CORRECTED: Use 2**m instead of 2**(m+1)
    F = lambda xi: distr.char((2**m) * np.pi * xi) 
    factor = (2**m) * np.pi

    p0 = distr.char(0)
    p1 = distr.char(factor)
    v0 = factor * distr.char_derivative(0)
    v1 = factor * distr.char_derivative(factor)
    a0 = factor**2 * distr.char_second_derivative(0)
    a1 = factor**2 * distr.char_second_derivative(factor)

    coeffs = hermite_spline_polynomial(p0, p1, v0, v1, a0, a1)

    powers = np.arange(6)
    s = lambda xi: np.sum(coeffs * np.power(xi[:, None], powers), axis=-1)

    ifft_input = np.concatenate((F(xis) - s(xis), np.zeros(N)))

    return ifft_input, coeffs

def meyer_coefficients(distr, nu, m, N):
    T2_xs, T2_coeffs = meyer_T2s(distr, m, N)
    T3_xs, T3_coeffs = meyer_T3s(distr, nu, m, N)

    ifft_input = np.vstack((T2_xs, T3_xs)) 
    ifft_output = np.fft.ifft(ifft_input)
    full_ifft_output = np.concatenate((ifft_output[:, 1:], ifft_output), axis=1)

    T2_ifft, T3_ifft = full_ifft_output

    ks = np.arange(-(3*N-1), 3*N)
    factor_2 = 2**(2+m/2) * np.pi
    factor_3 = 2**(1+m/2) * np.pi * np.exp(2j * np.pi * ks / 3)

    T2s = factor_2 * np.real(T2_ifft + I_QH(ks, T2_coeffs, 1j * 2 * np.pi / 3) / 3)
    T3s = factor_3 * (T3_ifft + I_QH(ks, T3_coeffs, 1j * 2 * np.pi / 3) / 3)
    
    cmks = 1 / (2 * np.pi) * (T2s + 2 * np.real(T3s))
    
    return cmks

def shannon_coefficients(distr, m, N):
    T_xs, T_coeffs = shannon_Ts(distr, m, N)

    ifft_output = np.fft.ifft(T_xs)
    T_ifft = np.concatenate((ifft_output[1:], ifft_output))

    ks = np.arange(-(2*N-1), 2*N)
    factor = 2**(2+m/2) * np.pi

    T2s = factor * np.real(T_ifft + I_QH(ks, T_coeffs, 1j * np.pi) / 2)
    
    cmks = 1 / (2 * np.pi) * (T2s)
    
    return cmks

def phi_hat_matrix(phi_hat_function, m, N, ws):
    ks = np.arange(-(3 * N-1), 3 * N) 
    pre_term = (1 / 2**(m/2)) * np.exp(-1j * (ws[:, np.newaxis] * ks[np.newaxis, :]) / (2**m))
    
    return pre_term * phi_hat_function(np.abs(ws) / (2**m))[:, np.newaxis]

def phi_hat_matrix_shannon(phi_hat_function, m, N, ws):
    ks = np.arange(-(2 * N-1), 2 * N) 
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
    result[cond2] = np.cos((np.pi / 2) * nu.function(x_for_nu))
    
    return result

def phi_hat_shannon(ws):
    abs_omega = np.abs(ws)

    result = np.zeros_like(abs_omega, dtype=float)
    
    cond1 = (abs_omega <= np.pi)
    result[cond1] = 1.0
    
    return result

def compute_convergence_errors_meyer(distr, nu, m, N, ws):
    cmks = meyer_coefficients(distr, nu, m, N)
    phi_hat_nu = lambda w : phi_hat_meyer(w, nu)
    matrix = phi_hat_matrix(phi_hat_nu, m, N, ws)

    weighted_matrix = matrix * cmks[np.newaxis, :]
    
    analytic_vals = distr.char(ws)
    errors = np.zeros(3 * N)
    
    # The center index of the array (corresponding to k=0)
    center_idx = 3 * N - 1
    
    # K = 0 (base case)
    projection = weighted_matrix[:, center_idx].copy()
    errors[0] = np.linalg.norm(analytic_vals - projection, 2)
    
    # Accumulate outwards for K > 0
    for K in range(1, 3 * N):
        projection += weighted_matrix[:, center_idx - K] + weighted_matrix[:, center_idx + K]
        errors[K] = np.linalg.norm(analytic_vals - projection, 2)
        
    return errors

def compute_convergence_errors_shannon(distr, m, N, ws):
    cmks = shannon_coefficients(distr, m, N)
    phi_hat_nu = lambda w : phi_hat_shannon(w)
    matrix = phi_hat_matrix_shannon(phi_hat_nu, m, N, ws)

    weighted_matrix = matrix * cmks[np.newaxis, :]
    
    analytic_vals = distr.char(ws)
    errors = np.zeros(2 * N)
    
    # The center index of the array (corresponding to k=0)
    center_idx = 2 * N - 1
    
    # K = 0 (base case)
    projection = weighted_matrix[:, center_idx].copy()
    errors[0] = np.linalg.norm(analytic_vals - projection, 2)
    
    # Accumulate outwards for K > 0
    for K in range(1, 2 * N):
        projection += weighted_matrix[:, center_idx - K] + weighted_matrix[:, center_idx + K]
        errors[K] = np.linalg.norm(analytic_vals - projection, 2)
        
    return errors

N_meyer = 2**12
N_shannon = N_meyer * 3 // 2
m = 5

w_max = 2**(m+1) * np.pi 
ws = np.linspace(-w_max, w_max, 1000)
distribution = NIG(alpha=5, beta=0, mu=0.3, delta=0.3)
# distribution = Gaussian(1, 0.5)

plt.figure(figsize=(10, 6))
for name, nu in [("Meyer Linear", nu_lin), ("Meyer Poly3", nu_poly3), ("Meyer Poly5", nu_poly5)]:
    print(f"Processing {name}...")
    errs = compute_convergence_errors_meyer(distribution, nu, m, N_meyer, ws)
    plt.semilogy(errs, label=name)

print(f"Processing Shannon...")
errs = compute_convergence_errors_shannon(distribution, m, N_shannon, ws)
plt.semilogy(errs, label="Shannon")

plt.title(f"Convergence Errors")
plt.xlabel("Number of terms (K)")
plt.ylabel("L2 Error")
plt.legend()
plt.grid(True, which="major", ls="-", alpha=0.5)
plt.show()