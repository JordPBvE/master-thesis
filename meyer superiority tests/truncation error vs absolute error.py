import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.special import factorial, gamma, kv

# --- Distributions ---
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

class Laplace(Distribution):
    def __init__(self, mu, b):
        phi = lambda ws: np.exp(-1j * mu * ws) / (1 + (b * ws)**2)
        
        # First derivative of ln(phi(ws))
        f_prime = lambda ws: -1j * mu - (2 * b**2 * ws) / (1 + (b * ws)**2)
        
        # Second derivative of ln(phi(ws))
        f_sec_prime = lambda ws: (2 * b**2 * ((b * ws)**2 - 1)) / (1 + (b * ws)**2)**2

        super().__init__(
            pdf = lambda ts: 1 / (2*b) * np.exp(-np.abs(ts - mu) / b),
            char = phi,
            char_deriv = lambda ws: f_prime(ws) * phi(ws),
            char_sec_deriv = lambda ws: (f_sec_prime(ws) + f_prime(ws)**2) * phi(ws)
        )

class VarianceGamma(Distribution):
    def __init__(self, mu, sigma, nu, theta):
        # Helper for the denominator base to keep equations clean
        # g(ws) = 1 + i*theta*nu*ws + 0.5*sigma^2*nu*ws^2
        g = lambda ws: 1 + 1j * theta * nu * ws + 0.5 * sigma**2 * nu * ws**2
        
        # Characteristic function (exp(-i * ws * x) convention)
        phi = lambda ws: np.exp(-1j * mu * ws) * (g(ws)**(-1/nu))
        
        # A(ws) represents the term: i*theta + sigma^2 * ws
        # This is derived from (1/nu) * g'(ws)
        A = lambda ws: 1j * theta + sigma**2 * ws
        
        # First derivative of ln(phi(ws))
        f_prime = lambda ws: -1j * mu - A(ws) / g(ws)
        
        # Second derivative of ln(phi(ws))
        f_sec_prime = lambda ws: -(sigma**2 * g(ws) - nu * A(ws)**2) / (g(ws)**2)

        # Variance Gamma PDF
        def pdf(ts):
            x = ts - mu
            # Prevent divide-by-zero exactly at x=0
            x_safe = np.where(x == 0, 1e-12, x)
            
            # Pre-compute constants for the Bessel function expression
            alpha = np.sqrt(theta**2 + 2 * sigma**2 / nu) / sigma**2
            beta = theta / sigma**2
            
            term1 = 2 * np.exp(beta * x_safe) / (nu**(1/nu) * np.sqrt(2 * np.pi) * sigma * gamma(1/nu))
            term2 = (np.abs(x_safe) / np.sqrt(2 * sigma**2 / nu + theta**2))**(1/nu - 0.5)
            term3 = kv(1/nu - 0.5, alpha * np.abs(x_safe))
            
            return term1 * term2 * term3

        super().__init__(
            pdf = pdf,
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
    
    numerator = (np.exp(a * k_column) * term1) - term0 
    denominator = np.power(a * k_column, ns + 1)
    sign = np.power(-1, ns)

    safe_result = np.sum(sign * (numerator / denominator), axis=1)

    k_zero_integral = np.sum((1 / np.arange(1, 7)) * coeffs)
    
    return np.where(k == 0, k_zero_integral, safe_result)

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

# --- OPTIMIZED COEFFICIENT CALCULATION ---
def shannon_coefficients(distr, m, N, k_max):
    T_xs, T_coeffs = shannon_Ts(distr, m, N)

    ifft_output = np.fft.ifft(T_xs)
    T_ifft = np.concatenate((ifft_output[1:], ifft_output))

    ks = np.arange(-(2*N-1), 2*N)
    factor = 2**(2+m/2) * np.pi

    T2s = factor * np.real(T_ifft + I_QH(ks, T_coeffs, 1j * np.pi) / 2)
    
    cmks = 1 / (2 * np.pi) * (T2s)
    
    return cmks[2*N-k_max - 1 : 2*N+k_max]

def meyer_coefficients(distr, m, N, k_max, nu):
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
    
    return cmks[3*N-k_max - 1: 3*N+k_max]

# --- OPTIMIZED PROJECTION FUNCTIONS ---

def get_meyer_projection(cmks_full, m, K, nu: Nu, ws):
    k_max = (len(cmks_full) - 1) // 2
    cmks = cmks_full[k_max - K : k_max + K + 1]
    ks = np.arange(-K, K + 1)

    ws_scaled = ws / (2**m)
    abs_ws_scaled = np.abs(ws_scaled)
    phi_hat_vec = np.zeros_like(ws, dtype=float)
    
    c1 = abs_ws_scaled <= (2*np.pi/3)
    phi_hat_vec[c1] = 1.0
    c2 = (abs_ws_scaled > (2*np.pi/3)) & (abs_ws_scaled <= (4*np.pi/3))
    phi_hat_vec[c2] = np.cos((np.pi/2) * nu.function((3/(2*np.pi))*abs_ws_scaled[c2] - 1))
    
    phi_hat_final = (1/2**(m/2)) * phi_hat_vec

    exponent_matrix = np.exp(-1j * (ws[:, np.newaxis] * ks[np.newaxis, :]) / (2**m))
    
    sum_term = np.dot(exponent_matrix, cmks)
    freq_domain = phi_hat_final * sum_term
    
    return np.array(freq_domain)

def get_shannon_projection(cmks_full, m, K, ws):
    k_max = (len(cmks_full) - 1) // 2
    cmks = cmks_full[k_max - K : k_max + K + 1]
    ks = np.arange(-K, K + 1)

    ws_scaled = ws / (2**m)
    abs_ws_scaled = np.abs(ws_scaled)
    phi_hat_vec = np.zeros_like(ws, dtype=float)
    
    c1 = abs_ws_scaled <= (np.pi)
    phi_hat_vec[c1] = 1.0
    
    phi_hat_final = (1/2**(m/2)) * phi_hat_vec

    exponent_matrix = np.exp(-1j * (ws[:, np.newaxis] * ks[np.newaxis, :]) / (2**m))
    
    sum_term = np.dot(exponent_matrix, cmks)
    freq_domain = phi_hat_final * sum_term
    
    return np.array(freq_domain)

# --- MAIN EXECUTION ---
m = 4
ws = np.linspace(-2**(m+2) * np.pi, 2**(m+2) * np.pi, 1000 * 2**m)
dw = ws[1] - ws[0]

mask_sh = np.abs(ws) <= 2**m * np.pi

# density = Gaussian(0, 0.1)
# density = NIG(5, 4.9, 0, 0.3)
density = VarianceGamma(0, .5, 2, 0)
# density = VarianceGamma(0, .1, 5, 0)
# density = Laplace(0, .1)
# density = Gaussian(1, 0.5)

fig, ax = plt.subplots(1, 3, figsize=(10, 5))

N=2**17

k_max_limit = N // 8
k_max = 1500
ks_coeffs = np.arange(-k_max_limit, k_max_limit+1)
Ks_to_plot = np.linspace(0, k_max, 50, dtype=int)


coeffs_shannon = shannon_coefficients(density, m, N, k_max_limit)
coeffs_meyer_x = meyer_coefficients(density, m, N, k_max_limit, nu_lin)
coeffs_meyer_c = meyer_coefficients(density, m, N, k_max_limit, nu_poly3)
coeffs_meyer_p = meyer_coefficients(density, m, N, k_max_limit, nu_poly5)


ax[0].plot(ks_coeffs, np.abs(coeffs_meyer_x), '--', label='Meyer Lin', alpha=0.7)
ax[0].plot(ks_coeffs, np.abs(coeffs_meyer_c), '--', label='Meyer Cos', alpha=0.7)
ax[0].plot(ks_coeffs, np.abs(coeffs_meyer_p), '--', label='Meyer Poly', alpha=0.7)
ax[0].plot(ks_coeffs, np.abs(coeffs_shannon), label='Shannon', alpha=0.8, linewidth=2)

ax[0].set_xscale('symlog', linthresh=1)
ax[0].set_yscale('log')

truncation_errs_sh = []
truncation_errs_mx = []
truncation_errs_m3 = []
truncation_errs_m5 = []

absolute_errs_sh = []
absolute_errs_mx = []
absolute_errs_m3 = []
absolute_errs_m5 = []

# --- 1. Pre-calculate invariant scaling functions (phi_hat) ---
ws_scaled = ws / (2**m)
abs_ws_scaled = np.abs(ws_scaled)

# Shannon
phi_hat_sh = np.zeros_like(ws, dtype=float)
phi_hat_sh[abs_ws_scaled <= np.pi] = 1.0
phi_hat_sh *= (1 / 2**(m/2))

# Meyer Helper
def get_meyer_phi_hat(nu):
    phi_hat = np.zeros_like(ws, dtype=float)
    c1 = abs_ws_scaled <= (2*np.pi/3)
    phi_hat[c1] = 1.0
    c2 = (abs_ws_scaled > (2*np.pi/3)) & (abs_ws_scaled <= (4*np.pi/3))
    phi_hat[c2] = np.cos((np.pi/2) * nu.function((3/(2*np.pi))*abs_ws_scaled[c2] - 1))
    return phi_hat * (1 / 2**(m/2))

phi_hat_mx = get_meyer_phi_hat(nu_lin)
phi_hat_m3 = get_meyer_phi_hat(nu_poly3)
phi_hat_m5 = get_meyer_phi_hat(nu_poly5)


# --- 2. Optimized Incremental Projector ---
ks_full = np.arange(-k_max_limit, k_max_limit + 1)

def compute_metrics(cmks, phi_hat, Ks_to_plot, ref_k_max):
    # Sort indices by magnitude to find the most significant coefficients
    sortmask = np.flip(np.argsort(np.abs(cmks)))
    
    # --- A. Compute Reference (Chunked to prevent memory spikes) ---
    ref_sum = np.zeros(len(ws), dtype=complex)
    ref_indices = sortmask[:2*ref_k_max + 1] 
    chunk_size = 2000 # Keeps matrix footprint under 100MB even for massive ws arrays
    
    for i in range(0, len(ref_indices), chunk_size):
        chunk = ref_indices[i:i+chunk_size]
        exp_chunk = np.exp(-1j * (ws[:, np.newaxis] * ks_full[chunk][np.newaxis, :]) / (2**m))
        ref_sum += exp_chunk @ cmks[chunk]
        
    reference_proj = phi_hat * ref_sum
    
    # --- B. Incrementally compute projections for the K loop ---
    current_sum = np.zeros(len(ws), dtype=complex)
    last_idx = 0
    
    trunc_errs = []
    abs_errs = []
    
    for K in Ks_to_plot:
        idx = 2 * K + 1 # Matching your original cumsum indexing logic
        
        # Calculate matrix only for the NEW terms added in this K-step
        if idx > last_idx:
            new_idx = sortmask[last_idx:idx]
            exp_chunk = np.exp(-1j * (ws[:, np.newaxis] * ks_full[new_idx][np.newaxis, :]) / (2**m))
            
            # Fast BLAS matrix-vector multiplication
            current_sum += exp_chunk @ cmks[new_idx]
            last_idx = idx
            
        proj = phi_hat * current_sum
        
        # Calculate and store errors
        trunc_errs.append(np.sqrt(dw) * np.linalg.norm(reference_proj - proj, 2))
        abs_errs.append(np.sqrt(dw) * np.linalg.norm(analytic - proj, 2) + np.abs(density.char(2**(m+2))))
    return trunc_errs, abs_errs


# --- 3. Execute ---
analytic = density.char(ws)
print(f"Calculating errors for {type(density).__name__}, m={m} over {len(Ks_to_plot)} K steps...")
print(f"cutoff error: {np.abs(density.char(2**(m+2)))}")

truncation_errs_sh, absolute_errs_sh = compute_metrics(coeffs_shannon, phi_hat_sh, Ks_to_plot, k_max_limit)
truncation_errs_mx, absolute_errs_mx = compute_metrics(coeffs_meyer_x, phi_hat_mx, Ks_to_plot, k_max_limit)
truncation_errs_m3, absolute_errs_m3 = compute_metrics(coeffs_meyer_c, phi_hat_m3, Ks_to_plot, k_max_limit)
truncation_errs_m5, absolute_errs_m5 = compute_metrics(coeffs_meyer_p, phi_hat_m5, Ks_to_plot, k_max_limit)


ax[1].plot(Ks_to_plot, truncation_errs_mx, '-', label='Meyer Lin', alpha=0.7)
ax[1].plot(Ks_to_plot, truncation_errs_m3, '-', label='Meyer Poly3', alpha=0.7)
ax[1].plot(Ks_to_plot, truncation_errs_m5, '-', label='Meyer Poly5', alpha=0.7)
ax[1].plot(Ks_to_plot, truncation_errs_sh, '-', label='Shannon', alpha=0.7)
ax[1].set_title("Truncation Error")
ax[1].set_xlabel("K")
ax[1].set_ylabel("L2 Error")
ax[1].set_yscale('log')
ax[1].grid(True, alpha=0.1)
ax[1].legend(fontsize='small')

ax[2].plot(Ks_to_plot, absolute_errs_mx, '-', label='Meyer Lin', alpha=0.7)
ax[2].plot(Ks_to_plot, absolute_errs_m3, '-', label='Meyer Cos', alpha=0.7)
ax[2].plot(Ks_to_plot, absolute_errs_m5, '-', label='Meyer Poly', alpha=0.7)
ax[2].plot(Ks_to_plot, absolute_errs_sh, '-', label='Shannon', alpha=0.7)
ax[2].set_title("Absolute error")
ax[2].set_xlabel("K")
ax[2].set_ylabel("L2 Error")
ax[2].set_yscale('log')
ax[2].grid(True, alpha=0.1)
ax[2].legend(fontsize='small')

plt.tight_layout()
plt.show()