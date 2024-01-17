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

def prix_instant_initial(S0, n, T, r, d, sigma, nb_simulations, K, type_payoff):
    H = 0
    for _ in range(nb_simulations):
        S = black_scholes(S0, n, T, r, d, sigma)
        H += payoff(S,K,type_payoff)
    esperance_payoff = H/nb_simulations
    return np.exp(-r*T)*esperance_payoff

S0 = 100
sigma = 0.1
r = 0.05
d = 0.03
K = 103
T = 1
n = 3
nb_simulations = [100*i for i in range(1,100)]

liste_prix = []
for nb_simu in tqdm(nb_simulations):
    liste_prix.append(prix_instant_initial(S0, n, T, r, d, sigma, nb_simu, K, "Call asiatique"))

print(liste_prix[-1])
plt.plot(nb_simulations, liste_prix)
plt.show()