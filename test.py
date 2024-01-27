import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm



def delta_t(n, T):
    """
    Pour n et T donnés, renvoie les t_i ainsi que l'incrément de temps dt
    """
    t = np.linspace(0, T, n)
    dt = t[1] - t[0]
    return t, dt


def mouvement_brownien(n, T, x0, r, sigma):
    """
    Renvoie une liste contenant les valeurs du mouvement brownien du temps 0 à T
    """
    t, dt = delta_t(n, T)
    B = np.zeros(n)
    B[0] = x0
    for i in range(1, n):
        B[i] = B[i-1] + np.random.normal(loc=r*dt, scale=sigma*np.sqrt(dt))
    return B

def mouvement_brownien_standard(n, T):
    return mouvement_brownien(n, T, 0, 0, 1)



def black_scholes(S0, n, T, r, d, sigma, renvoie_brownien=False):
    """
    Renvoie une liste contenant l'évolution du prix d'un actif sous-jacent suivant le modèle de Black-Scholes
    Peut aussi renvoyer le mouvement brownien correspondant
    """
    temps, dt = delta_t(n, T)
    W = mouvement_brownien_standard(n, T)
    S = []
    for t, Wt in zip(temps, W):
        S.append(S0*np.exp((r-d)*t - sigma**2*t/2 + sigma*Wt))
    if renvoie_brownien:
        return S, W
    return S


def payoff(S, K, type_payoff):
    if type_payoff == "Call EU":
        return max(S[-1]-K, 0)
    elif type_payoff == "Call asiatique":
        return max(np.mean(S)-K, 0)
    else:
        raise ValueError

def prix_instant_initial(S0, n, T, r, d, sigma, nb_simulations, K, type_payoff, frequence_mesure=1000):
    H = 0
    prix = []
    iterations = []
    for i in tqdm(range(1,nb_simulations+1)):
        S = black_scholes(S0, n, T, r, d, sigma)
        H += payoff(S,K,type_payoff)
        if i % frequence_mesure == 0:
            esperance_payoff = H/i
            prix.append(np.exp(-r*T)*esperance_payoff)
            iterations.append(i)
    return prix, iterations

S0 = 100
sigma = 0.1
r = 0.05
d = 0.03
K = 103
T = 1
n = 3
nb_simulations = 1_000_000

result, frequence  = prix_instant_initial(S0, n, T, r, d, sigma, nb_simulations, K, "Call asiatique")

print(result[-1])
# 1.2564532648137718 avec 1 million de simulations
plt.plot(frequence, result)
plt.show()