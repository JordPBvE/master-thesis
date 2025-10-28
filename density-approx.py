import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfi
from scipy import integrate
import scipy.stats as stats

def phi_meyer(ts):
    numerator = np.sin(2 * np.pi * ts / 3) + (4 / 3) * ts * np.cos(4 * np.pi * ts / 3)
    denominator = np.pi * ts - (16 * np.pi / 9) * ts**3
    
    result = np.where(
        np.isclose(ts, 0),
        (2/3 + 4 / (3 * np.pi)),
        numerator / denominator
    )
    return result

def gauss_meyer_coefficients(sigma, mu, m, k_max):
    ks = np.arange(-k_max, k_max + 1) 
    factor = 1j / 2**(1 + m / 2)

    def G(a, b, x):
        term1 = 0.5
        term2 = np.sqrt(np.pi / a)
        term3 = np.exp(-b**2 / (4 * a))
        term4_input = np.sqrt(a) * x + b / np.sqrt(a)
        term4 = erfi(term4_input)
        return term1 * term2 * term3 * term4

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

def project(cmks, scalingfunction, m, ts):
    k_max = (cmks.size - 1) // 2
    ks = np.arange(-k_max, k_max + 1)

    args = (2**m * ts[:, np.newaxis]) - ks[np.newaxis, :]
    phi_matrix = 2**(m/2) * scalingfunction(args) 
    
    projection = np.dot(phi_matrix, cmks)

    return np.real(projection)

k_max = 100
m = 5
ts = np.linspace(-15, 15, 300)

mu = 0
sigma = 1

cmks = gauss_meyer_coefficients(sigma, mu, m, k_max)
projection =  project(cmks, phi_meyer, m, ts)

print(f"integral = {integrate.trapezoid(projection, ts)}")


# 3. Create the plot
plt.figure(figsize=(10, 6))
plt.plot(ts, projection, 'k', linewidth=2, label='Meyer Wavelet approximation')
plt.plot(ts, stats.norm.pdf(ts, mu, sigma), 'r-.', linewidth=2, label = 'analytic standard normal')
plt.title("Meyer Scaling Function")
plt.xlabel("t")
plt.ylabel("$\phi(t)$")
plt.grid(True)
plt.legend()
plt.show()