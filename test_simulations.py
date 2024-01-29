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


def payoff(S, K, type_payoff, T=1, frequence_barriere="M", valeur_barriere=80):
    match type_payoff:
        case "Call EU":
            return max(S[-1]-K, 0)
        case "Call asiatique":
            return max(np.mean(S)-K, 0)
        case "Call down and out":
            n = len(S)
            match frequence_barriere:
                case "M":
                    k = n//(12*T)
                    i = k
                    while i < n:
                        if S[i] < valeur_barriere:
                            # print("BARRIERE")
                            return 0.0
                        i += k
                    return max(S[-1]-K, 0)
        case "Call down and in":
            n = len(S)
            match frequence_barriere:
                case "M":
                    k = n//(12*T)
                    i = k
                    while i < n:
                        if S[i] < valeur_barriere:
                            # print("BARRIERE")
                            return max(S[-1]-K, 0)
                        i += k
                    return 0.0
        case _:
            raise ValueError("Mauvais type de payoff")

def prix_instant_initial(S0, n, T, r, d, sigma, nb_simulations, K, type_payoff, frequence_mesure=1000,frequence_barriere="M", valeur_barriere=80):
    H = 0
    prix = []
    iterations = []
    for i in tqdm(range(1,nb_simulations+1)):
        S = black_scholes(S0, n, T, r, d, sigma)
        H += payoff(S,K,type_payoff,T=T, frequence_barriere=frequence_barriere, valeur_barriere=valeur_barriere)
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
n = 12
nb_simulations = 10**5
H = 80
frequence_barriere = "M"


result, frequence  = prix_instant_initial(S0, n, T, r, d, sigma,
                    nb_simulations, K, "Call down and in",
                    frequence_mesure=1,
                    frequence_barriere=frequence_barriere,
                    valeur_barriere=H)

print(result[-1])
# 1.2564532648137718 avec 1 million de simulations, Call Asiatique
plt.plot(frequence, result)
plt.show()