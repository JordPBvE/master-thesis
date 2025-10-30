import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfi, erf
from scipy import integrate
from time import time 
import scipy.stats as stats

def phi_shannon(ts):
    return np.sinc(ts)

def phi_meyer(ts):
    numerator = np.sin(2 * np.pi * ts / 3) + (4 / 3) * ts * np.cos(4 * np.pi * ts / 3)
    denominator = np.pi * ts - (16 * np.pi / 9) * ts**3
    
    result = np.where(
        np.isclose(ts, 0),
        (2/3 + 4 / (3 * np.pi)),
        numerator / denominator
    )
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

    return factor * (func(2**m * np.pi) - func(-2**m * np.pi))

def gauss_meyer_coefficients(sigma, mu, m, k_max):
    ks = np.arange(-k_max, k_max + 1) 
    factor = 1j / 2**(1 + m / 2)

    order_2_term = -0.5 * (sigma**2) + 0j

    order_1_term_p = -1j * (mu - (ks / 2**m) + (3 / 2**(m + 2)))
    order_1_term_m = -1j * (mu - (ks / 2**m) - (3 / 2**(m + 2)))
    order_1_term = -1j * (mu - (ks / 2**m))

    T1_f = lambda x: G(order_2_term, order_1_term_m, x) - G(order_2_term, order_1_term_p, x)
    T2_f = lambda x: G(order_2_term, order_1_term, x)
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

ms = [1, 2, 3, 4, 5]
Ks = 4 * np.power(2, ms)
ts = np.linspace(-5, 5, 300)

mu = 0
sigma = 1/5

repeat = 20

print("SHANNON")
for m, K in zip(ms, Ks):
    start = time()
    for _ in range(repeat):
        cmks = gauss_shannon_coefficients(sigma, mu, m, K)
        projection =  project(cmks, phi_shannon, m, K, ts)
    stop = time()

    analytical = stats.norm.pdf(ts, mu, sigma)
    err = np.max(np.abs(projection - analytical))
    duration = 1000*(stop-start)/repeat

    print(f'm={m}, K={K}, err={"{:.2E}".format(err)}, time={round(duration, 2)} ms')

print("MEYER")
for m, K in zip(ms, Ks):
    start = time()
    for _ in range(repeat):
        cmks = gauss_meyer_coefficients(sigma, mu, m, K)
        projection =  project(cmks, phi_meyer, m, K, ts)
    stop = time()

    analytical = stats.norm.pdf(ts, mu, sigma)
    err = np.max(np.abs(projection - analytical))
    duration = 1000*(stop-start)/repeat

    print(f'm={m}, K={K}, err={"{:.2E}".format(err)}, time={round(duration, 2)} ms')
