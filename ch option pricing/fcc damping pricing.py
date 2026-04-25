import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
from enum import Enum
from scipy.stats import norm
from scipy.special import kv, gamma
from scipy.fft import dct
from math import ceil
import scipy.linalg as la

class OptionVariant(Enum):
    PUT = 'PUT'
    CALL = 'CALL'
PUT = OptionVariant.PUT
CALL = OptionVariant.CALL

class Option:
    def __init__(self, var : OptionVariant, strike, damping):
        self.var = var
        self.K = strike
        self.R = damping
    
    def hat_v_T(self, xi):
        if self.var == OptionVariant.CALL:
            return self.K / ((self.R + 1j * xi) * (self.R + 1j * xi - 1))
        if self.var == OptionVariant.PUT:
            return 0

class Distribution:
    def __init__(self, pdf, char):
        self.pdf = pdf
        self.char = char

class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        phi = lambda ws: np.exp(-1j * mu * ws - 0.5 * (sigma**2) * (ws**2))
        super().__init__(
            pdf = lambda ts: stats.norm.pdf(ts, mu, sigma),
            char = phi
        )
        self.mu = mu
        self.sigma = sigma

class NIG(Distribution):
    def __init__(self, alpha, beta, mu, delta):
        gamma = np.sqrt(alpha**2 - beta**2)   
        phi = lambda ws: np.exp(-mu * 1j * ws + delta * (gamma - np.sqrt(alpha**2 - (beta - 1j * ws)**2)))

        super().__init__(
            pdf = lambda ts: stats.norminvgauss.pdf(ts, a=alpha, b=beta, loc=mu, scale=delta),
            char = phi
        )

class Uniform(Distribution):
    def __init__(self, a, b):
        def phi(ws):
            ws = np.asarray(ws)
            # Mask zeros to prevent divide-by-zero warnings
            safe_ws = np.where(ws == 0, 1.0, ws)
            res = 1j * (np.exp(-1j * safe_ws * b) - np.exp(-1j * safe_ws * a)) / (safe_ws * (b - a))
            return np.where(ws == 0, 1.0 + 0j, res)

        super().__init__(
            pdf = lambda ts: stats.uniform.pdf(ts, loc=a, scale=b-a),
            char = phi
        )

class Laplace(Distribution):
    def __init__(self, mu, b):
        phi = lambda ws: np.exp(-1j * mu * ws) / (1 + (b * ws)**2)
        super().__init__(
            pdf = lambda ts: 1 / (2*b) * np.exp(-np.abs(ts - mu) / b),
            char = phi
        )

class VarianceGamma(Distribution):
    def __init__(self, mu, sigma, nu, theta):
        g = lambda ws: 1 + 1j * theta * nu * ws + 0.5 * sigma**2 * nu * ws**2
        
        phi = lambda ws: np.exp(-1j * mu * ws) * (g(ws)**(-1/nu))
        
        # Variance Gamma PDF
        def pdf(ts):
            x = ts - mu
            x_safe = np.where(x == 0, 1e-12, x)
            
            alpha = np.sqrt(theta**2 + 2 * sigma**2 / nu) / sigma**2
            beta = theta / sigma**2
            
            term1 = 2 * np.exp(beta * x_safe) / (nu**(1/nu) * np.sqrt(2 * np.pi) * sigma * gamma(1/nu))
            term2 = (np.abs(x_safe) / np.sqrt(2 * sigma**2 / nu + theta**2))**(1/nu - 0.5)
            term3 = kv(1/nu - 0.5, alpha * np.abs(x_safe))
            
            return term1 * term2 * term3

        super().__init__(
            pdf = pdf,
            char = phi
        )

class Damped(Distribution):
    def __init__(self, distr : Distribution, R):
        super().__init__(
            pdf = lambda xs : np.exp(R * xs) * distr.pdf(xs),
            char = lambda xis : distr.char(xis + 1j * R)
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

def chebyshev_nodes(N):
    return np.cos(np.arange(N+1) * np.pi / N)

def meyer_T2s(distr : Distribution, m, N):
    nodes = chebyshev_nodes(N)
    tau = lambda xi: distr.char(2**m * np.pi * (xi + 1) / 3) 

    return tau(nodes)

def meyer_T3s(distr : Distribution, nu : Nu, m, N):
    nodes = chebyshev_nodes(N)
    tau = lambda xi: distr.char(2**m * np.pi * (xi + 3) / 3) * np.cos((np.pi/2) * nu.function((xi+1) / 2))

    return tau(nodes)

def meyer_t2s(option : Option, m, N):
    nodes = chebyshev_nodes(N)
    tau = lambda xi: option.hat_v_T(2**m * np.pi * (xi + 1) / 3) 

    return tau(nodes)

def meyer_t3s(option : Option, nu : Nu, m, N):
    nodes = chebyshev_nodes(N)
    tau = lambda xi: option.hat_v_T(2**m * np.pi * (xi + 3) / 3) * np.cos((np.pi/2) * nu.function((xi+1) / 2))

    return tau(nodes)

def shannon_Ts(distr : Distribution, m, N):
    nodes = chebyshev_nodes(N)
    tau = lambda xi: distr.char(2**(m-1) * np.pi * (xi + 1)) 

    return tau(nodes)

def shannon_ts(option : Option, m, N):
    nodes = chebyshev_nodes(N)
    tau = lambda xi: option.hat_v_T(2**(m-1) * np.pi * (xi + 1)) 

    return tau(nodes)
    
def chebyshev_weights(k_max, N, k_to_kappa):
    "outputs matrix with as rows w_n(-k_max) ... w_n(k_max)"
    # compute the matrices for K > 0, rest can be deduced
    ks = np.arange(1, k_max + 1, dtype=float)
    kappas = k_to_kappa(ks)             # size k_max 
    Omega_pos = np.zeros((N+1, k_max), dtype=complex)

    def gamma():
        multiplier = 2 / kappas
        Gamma = np.zeros((8 * N//7, len(kappas)), dtype=complex)
        Gamma[0::2, :] = multiplier * np.sin(kappas)
        Gamma[1::2, :] = -1j * multiplier * np.cos(kappas)

        return Gamma
 
    def rho_phase_1():
        Rho = np.zeros((N+1, k_max), dtype=complex)
        # phase 1 
        Rho[1, :] = Gamma[0, :]
        Rho[2, :] = 2 * Gamma[1, :] - (2 / (1j * kappas)) * Gamma[0, :]

        for n in range(2, N):
            Rho[n + 1, :] = 2 * Gamma[n, :] - (2 * n / (1j * kappas)) * Rho[n, :] + Rho[n-1, :]

        mask = np.arange(N+1)[:, None] < np.ceil(kappas)  # n < kappa condition
        return np.where(mask, Rho, 0j)
    
    Gamma = gamma()
    Rho = rho_phase_1()

    # phase 2
    M = N * 4 // 7

    def get_rho_2M():
        two_M = 2 * M
        M2 = M**2
        k2 = kappas**2
        
        p0 = 1 / two_M
        p1 = kappas / (two_M**3)
        p2 = 3 * k2 / (two_M**5)
        p3 = (15 * k2 - 4 * M2) * kappas / (two_M**7)
        p4 = (105 * k2 - 60 * M2) * k2 / (two_M**9)
        p5 = (945 * kappas**4 - 840 * k2 * M2 + 16 * M**4) * kappas / (two_M**11)
        p6 = (-12600 * k2 * M2 + 1008 * M**4 + 10395 * kappas**4) * k2 / (two_M**13)
        
        sum_even = p0 - p2 + p4 - p6
        sum_odd  = p1 - p3 + p5
        
        return 2j * (sum_even * np.sin(kappas) + sum_odd * np.cos(kappas))

    rho_2M = get_rho_2M()

    for i, kappa in enumerate(kappas):
        n0 = ceil(kappa)
        if n0 <= N:
            A = np.zeros((3, 2 * M - n0), dtype=complex)
            A[0, 1:] = np.ones(2 * M - n0 - 1)                        
            A[1, :] = 2 * np.arange(n0, 2*M) / (1j * kappa)
            A[2, :-1] = -np.ones(2 * M - n0 - 1)

            b = 2 * Gamma[n0:2*M, i]
            b[0]  += Rho[n0-1, i]
            b[-1] -= rho_2M[i]

            rho_M = la.solve_banded((1, 1), A, b)
            Rho[n0:, i] = rho_M[:N+1-n0]
    
    Omega_pos[0, :] = Rho[1, :]
    for n in range(1, N+1):
        Omega_pos[n, :] = Gamma[n, :] - (n / (1j * kappas)) * Rho[n, :]

    ns = np.arange(N + 1)
    Omega_0 = np.zeros((N + 1, 1), dtype=complex)
    even_mask = (ns % 2 == 0)
    Omega_0[even_mask, 0] = 2 / (1 - ns[even_mask]**2)
    
    # Stack horizontally: [-k_max, ..., -1, 0, 1, ..., k_max]
    Omega_full = np.hstack((np.fliplr(np.conj(Omega_pos)), Omega_0, Omega_pos))

    return Omega_full

# --- OPTIMIZED COEFFICIENT CALCULATION ---
def shannon_coefficients(distr, option, m, N, k_max):
    T_xs = shannon_Ts(distr, m, N)
    t_xs = shannon_ts(option, m, N)

    input = np.vstack((T_xs, t_xs)) 
    output = dct(input, type=1) / N # size (2, 2*N)
    output[:, 0] /= 2
    output[:, -1] /= 2

    weight_matrix = chebyshev_weights(k_max, N, lambda k : k * np.pi / 2) # size(N, 2*k_max+1)

    Ts_ints, ts_ints = output @ weight_matrix

    ks = np.arange(-k_max, k_max + 1)
    factor2 = 2**(m/2) * np.pi
    exponent2 = np.exp(1j * np.pi * ks / 2) 

    Ts = factor2 * np.real(exponent2 * Ts_ints)
    ts = factor2 * np.real(exponent2 * ts_ints)

    cmks = 1 / (2 * np.pi) * Ts
    Vmks = 1 / (2 * np.pi) * ts

    return cmks, Vmks # size (2 * k_max_limit + 1)

def meyer_coefficients(distr : Distribution, option : Option, m, N, k_max, nu):
    T2_xs = meyer_T2s(distr, m, N)
    T3_xs = meyer_T3s(distr, nu, m, N)
    t2_xs = meyer_t2s(option, m, N)
    t3_xs = meyer_t3s(option, nu, m, N)

    input = np.vstack((T2_xs, T3_xs, t2_xs, t3_xs)) 
    output = dct(input, type=1) / N
    output[:, 0] /= 2
    output[:, -1] /= 2

    weight_matrix = chebyshev_weights(k_max, N, lambda k : k * np.pi / 3)

    T2s_ints, T3s_ints, t2s_ints, t3s_ints = output @ weight_matrix

    ks = np.arange(-k_max, k_max + 1)
    factor2 = 2**(m/2 + 1) * np.pi / 3
    exponent2 = np.exp(1j * np.pi * ks / 3) 
    factor3 = 2**(m/2) * np.pi * np.exp(1j * ks * np.pi) / 3

    T2s = factor2 * np.real(exponent2 * T2s_ints)
    t2s = factor2 * np.real(exponent2 * t2s_ints)

    T3s = factor3 * T3s_ints
    t3s = factor3 * t3s_ints

    cmks = 1 / (2 * np.pi) * (T2s + 2 * np.real(T3s))
    Vmks = 1 / (2 * np.pi) * (t2s + 2 * np.real(t3s))

    return cmks, Vmks # size (2 * k_max_limit + 1)

def gaussian_price(log_distr_T: Gaussian, option, r, t, T):
    def analytic_value(sigma, r, t, T, St, strike):
        tau = T - t
        d1 = (np.log(St/strike) + (r+sigma**2/2)*tau) / (sigma*np.sqrt(tau))
        d2 = d1 - sigma*np.sqrt(tau)

        if option.var is CALL: return St * norm.cdf(d1) - strike * np.exp(-r*tau) * norm.cdf(d2)
        if option.var is PUT:  return strike * np.exp(-r*tau) * norm.cdf(-d2) - St * norm.cdf(-d1)

    strike = option.K
    tau = T - t
    sigma = log_distr_T.sigma / np.sqrt(tau)
    St = strike * np.exp(log_distr_T.mu - (r - sigma**2/2)*tau )

    return analytic_value(sigma, r, t, T, St, strike)

r = 0.05

T = 1
t = 0

option = Option(CALL, strike=100, damping=1.2)

log_distr_at_T = Gaussian(0, 0.1)

damped_distr = Damped(log_distr_at_T, option.R)

N = 2**12
k_max_limit = 2**10
m = 3

k_max = 40
tau = T - t

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

cmks_sh, Vmks_sh = shannon_coefficients(damped_distr, option, m, N, k_max_limit)
cmks_mx, Vmks_mx = meyer_coefficients(damped_distr, option, m, N, k_max_limit, nu_lin)
cmks_mc, Vmks_mc = meyer_coefficients(damped_distr, option, m, N, k_max_limit, nu_poly3)
cmks_mp, Vmks_mp = meyer_coefficients(damped_distr, option, m, N, k_max_limit, nu_poly5)

ax[0].plot(np.arange(-k_max_limit, k_max_limit+1), np.abs(cmks_mx * Vmks_mx), '--', label='Meyer Lin', alpha=0.7)
ax[0].plot(np.arange(-k_max_limit, k_max_limit+1), np.abs(cmks_mc * Vmks_mc), '--', label='Meyer Cos', alpha=0.7)
ax[0].plot(np.arange(-k_max_limit, k_max_limit+1), np.abs(cmks_mp * Vmks_mp), '--', label='Meyer Poly', alpha=0.7)
ax[0].plot(np.arange(-k_max_limit, k_max_limit+1), np.abs(cmks_sh * Vmks_sh), label='Shannon', alpha=0.8, linewidth=2)
ax[0].set_title(r"$|c_{m,k} \cdot V_{m,k}|$")
ax[0].set_xscale('symlog', linthresh=1)
ax[0].set_yscale('log')

def coeffs_to_errors(cmks, Vmks):
    vs = np.exp(-r*tau) * cmks * Vmks
    sortmask = np.flip(np.argsort(np.abs(vs)))
    cumulative_sums = np.cumsum(vs[sortmask])

    if isinstance(log_distr_at_T, Gaussian):
        reference = gaussian_price(log_distr_at_T, option, r, t, T)
        print(f"reference price: {reference}")
    else:
        reference = cumulative_sums[-1]
    print(f"projection: {cumulative_sums[-1]}")

    errors = np.abs(cumulative_sums[: k_max * 2: 2] - reference)
    return errors

errs_sh = coeffs_to_errors(cmks_sh , Vmks_sh)      
errs_mx = coeffs_to_errors(cmks_mx , Vmks_mx)
errs_mc = coeffs_to_errors(cmks_mc , Vmks_mc)
errs_mp = coeffs_to_errors(cmks_mp , Vmks_mp)

ax[1].plot(np.arange(k_max), errs_mx, label="Meyer Linear")
ax[1].plot(np.arange(k_max), errs_mc, label="Meyer Cos")
ax[1].plot(np.arange(k_max), errs_mp, label="Meyer Poly")
ax[1].plot(np.arange(k_max), errs_sh, label="Shannon")

ax[1].set_yscale('log')

ax[1].set_title(f"Absolute Error (m={m}, N=2^{int(np.log2(N))})")
ax[1].set_xlabel("Number of terms")
ax[1].set_ylabel("Error")
ax[1].legend()
ax[1].grid(True, which="both", ls="-", alpha=0.5)
plt.show()