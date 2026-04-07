import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.stats as stats
import scipy.special as sp

# --- Distributions ---
class Distribution:
    def __init__(self, f, f_hat):
        self.f = f
        self.f_hat = f_hat

class Gaussian(Distribution):
    def __init__(self, mu, sigma):
        self.f = lambda x : 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))
        self.f_hat = lambda w : np.exp(-1j * mu * w - 0.5 * sigma**2 * w**2)

class Cauchy(Distribution):
    def __init__(self, mu, theta):
        self.f = lambda x : (1/np.pi) * theta / ((x-mu)**2 + theta**2)
        self.f_hat = lambda w : np.exp(-1j * mu * w - theta * np.abs(w))

class Laplace(Distribution):
    def __init__(self, mu, b):
        self.f = lambda x : 1/(2*b) * np.exp(-np.abs(x-mu)/b)
        self.f_hat = lambda w : np.exp(-1j * mu * w) / (1 + b**2 * w**2)

class NormalInverseGaussian(Distribution):
    def __init__(self, a=1.0, b=0.0, loc=0.0, scale=1.0):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale
        
        # Built-in scipy PDF
        self.f = lambda x: stats.norminvgauss.pdf(x, a, b, loc=loc, scale=scale)
        
        # Characteristic function phi(-w) to match the f_hat convention
        def cf_neg_w(w):
            alpha = a / scale
            beta = b / scale
            delta = scale
            mu = loc
            
            gamma = np.sqrt(alpha**2 - beta**2)
            term = np.sqrt(alpha**2 - (beta - 1j*w)**2)
            return np.exp(-1j * mu * w + delta * (gamma - term))
        
        self.f_hat = cf_neg_w

class VarianceGamma(Distribution):
    def __init__(self, mu=0.0, sigma=1.0, theta=0.0, nu=0.5):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        
        def pdf(x):
            # Small offset to prevent log(0) or div by zero at exactly x=mu
            x_adj = np.where(x == mu, x + 1e-12, x)
            abs_diff = np.abs(x_adj - mu)
            
            c = np.sqrt(2.0 / nu + (theta / sigma)**2)
            
            term1 = np.sqrt(2.0) * np.exp((theta * (x_adj - mu)) / sigma**2)
            term2 = sigma * np.sqrt(np.pi) * (nu**(1.0/nu)) * sp.gamma(1.0/nu)
            
            pow_val = 1.0/nu - 0.5
            term3 = (abs_diff / (sigma * c))**pow_val
            
            # Modified Bessel function of the second kind
            term4 = sp.kv(pow_val, (c * abs_diff) / sigma)
            
            return (term1 / term2) * term3 * term4

        self.f = pdf
        self.f_hat = lambda w: np.exp(-1j * mu * w) * (1 + 1j * theta * nu * w + 0.5 * sigma**2 * nu * w**2)**(-1.0/nu)

# --- Meyer Nu Functions ---
def nu_lin(x):
    func = lambda t: t
    return np.piecewise(x, [x <= 0, x >= 1], [0, 1, func])

def nu_poly(x):
    func = lambda t: t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
    return np.piecewise(x, [x <= 0, x >= 1], [0, 1, func])

def nu_cos(x):
    func = lambda t: (1- np.cos(np.pi * t))/2
    return np.piecewise(x, [x <= 0, x >= 1], [0, 1, func])

# --- OPTIMIZED COEFFICIENT CALCULATION ---
def precompute_shannon_coeffs(dist, m, k_max):
    """Integrates once for all k in [-k_max, k_max]"""
    ks = np.arange(-k_max, k_max + 1)
    factor = 1/(2**(m/2+1)*np.pi)
    bound = 2**(m)*np.pi
    
    cmks = np.zeros_like(ks, dtype=float)
    
    print(f"  Integrating Shannon coeffs (m={m})...")
    for i, k in enumerate(ks):
        func = lambda w: np.real(dist.f_hat(w) * np.exp(1j * (k / 2**m) * w))
        T, _ = integrate.quad(func, -bound, bound, limit=10000, epsabs=1.49 * 10**(-13), epsrel=1.49 * 10**(-13))
        cmks[i] = factor * T
    return cmks

def precompute_meyer_coeffs(dist, m, k_max, nu_func):
    """Integrates once for all k in [-k_max, k_max] using Meyer splitting"""
    ks = np.arange(-k_max, k_max + 1)
    factor = 1/(2**(m/2+1)*np.pi)
    a = 2**(m+1)*np.pi/3
    b = 2**(m+2)*np.pi/3
    
    cmks = np.zeros_like(ks, dtype=float)
    
    print(f"  Integrating Meyer coeffs (m={m})...")
    for i, k in enumerate(ks):
        f1_base = lambda w: dist.f_hat(w) * np.exp(1j * (k / 2**m) * w)
        
        func_T2 = lambda w: np.real(f1_base(w))
        func_T3 = lambda w: np.real(f1_base(w) * np.cos((np.pi/2) * nu_func(3/(2**(m+1)*np.pi) * w - 1)))

        T2, _ = integrate.quad(func_T2, 0, a, limit=10000, epsabs=1.49 * 10**(-15), epsrel=1.49 * 10**(-15))
        T3, _ = integrate.quad(func_T3, a, b, limit=10000, epsabs=1.49 * 10**(-15), epsrel=1.49 * 10**(-15))
        
        cmks[i] = factor * 2 * (T2 + T3) # both already set to real
        
    return cmks

# --- OPTIMIZED PROJECTION FUNCTIONS ---

def get_meyer_projection(cmks_full, m, K, nu_func, ws):
    k_max = (len(cmks_full) - 1) // 2
    cmks = cmks_full[k_max - K : k_max + K + 1]
    ks = np.arange(-K, K + 1)

    ws_scaled = ws / (2**m)
    abs_ws_scaled = np.abs(ws_scaled)
    phi_hat_vec = np.zeros_like(ws, dtype=float)
    
    c1 = abs_ws_scaled <= (2*np.pi/3)
    phi_hat_vec[c1] = 1.0
    c2 = (abs_ws_scaled > (2*np.pi/3)) & (abs_ws_scaled <= (4*np.pi/3))
    phi_hat_vec[c2] = np.cos((np.pi/2) * nu_func((3/(2*np.pi))*abs_ws_scaled[c2] - 1))
    
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

m = 2
ws = np.linspace(-2**(m+2) * np.pi, 2**(m+2) * np.pi, 10000)

density = Cauchy(0, 0.5)
fig, ax = plt.subplots(1, 2, figsize=(10, 7))

analytic = density.f_hat(ws)
ax[0].plot(ws, analytic, '-', lw=2)
ax[0].set_title(f'{type(density).__name__} Analytic')
ax[0].set_xlabel("w")
ax[0].grid(True, alpha=0.1)

k_max_limit = 600
Ks_to_plot = np.linspace(0, k_max_limit, 50, dtype=int)

coeffs_shannon = precompute_shannon_coeffs(density, m, k_max_limit)
coeffs_meyer_x = precompute_meyer_coeffs(density, m, k_max_limit, nu_lin)
coeffs_meyer_c = precompute_meyer_coeffs(density, m, k_max_limit, nu_cos)
coeffs_meyer_p = precompute_meyer_coeffs(density, m, k_max_limit, nu_poly)

errs_sh = []
errs_mx = []
errs_mc = []
errs_mp = []

print(f"Calculating errors for {type(density).__name__}, m={m} over {len(Ks_to_plot)} K steps...")

for K in Ks_to_plot:        
    proj_sh = get_shannon_projection(coeffs_shannon, m, K, ws)
    errs_sh.append(np.linalg.norm(analytic - proj_sh, 2))

    proj_mx = get_meyer_projection(coeffs_meyer_x, m, K, nu_lin, ws)
    errs_mx.append(np.linalg.norm(analytic - proj_mx, 2))

    proj_mc = get_meyer_projection(coeffs_meyer_c, m, K, nu_cos, ws)
    errs_mc.append(np.linalg.norm(analytic - proj_mc, 2))
    
    proj_mp = get_meyer_projection(coeffs_meyer_p, m, K, nu_poly, ws)
    errs_mp.append(np.linalg.norm(analytic - proj_mp, 2))

ax[1].plot(Ks_to_plot, errs_mx, '-', label='Meyer Lin', alpha=0.7)
ax[1].plot(Ks_to_plot, errs_mc, '-', label='Meyer Cos', alpha=0.7)
ax[1].plot(Ks_to_plot, errs_mp, '-', label='Meyer Poly', alpha=0.7)
ax[1].plot(Ks_to_plot, errs_sh, '-', label='Shannon', alpha=0.7)

ax[1].set_title(f"m = {m}")
ax[1].set_xlabel("K")
ax[1].set_ylabel("L2 Error")
ax[1].set_yscale('log')
ax[1].grid(True, alpha=0.1)
ax[1].legend(fontsize='small')

plt.tight_layout()
plt.show()