import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import qmc
import scipy.stats as stats



def delta_t(n, T):
    """
    Pour n et T donnés, renvoie les t_i ainsi que l'incrément de temps dt
    """
    t = np.linspace(0, T, n)
    dt = t[1] - t[0]
    return t, dt


def mouvement_brownien(n, T, x0, r, sigma, methode="Pseudo"):
    """
    Renvoie une liste contenant les valeurs du mouvement brownien du temps 0 à T
    """
    t, dt = delta_t(n, T)
    B = np.zeros(n)
    B[0] = x0
    if methode == "Quasi":
        distrib = halton_norm(n-1)
    for i in range(1, n):
        if methode == "Pseudo":
            # B[i] = B[i-1] + np.random.normal(loc=r*dt, scale=sigma*np.sqrt(dt))
            B[i] = B[i-1] + (r*dt + sigma*np.sqrt(dt)*np.random.normal(loc=0, scale=1))
        elif methode == "Quasi":
            print(float(distrib[i-1]))
            B[i] = B[i-1] + (r*dt + sigma*np.sqrt(dt)*float(distrib[i-1]))
        else:
            raise Exception("Mauvaise méthode !")
    return B

def mouvement_brownien_standard(n, T, methode="Pseudo"):
    return mouvement_brownien(n, T, 0, 0, 1, methode)


def black_scholes(S0, n, T, r, d, sigma, methode="Pseudo", renvoie_brownien=False):
    """
    Renvoie une liste contenant l'évolution du prix d'un actif sous-jacent suivant le modèle de Black-Scholes
    Peut aussi renvoyer le mouvement brownien correspondant
    """
    temps, dt = delta_t(n, T)
    W = mouvement_brownien_standard(n, T, methode)
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

def prix_instant_initial(S0, n, T, r, d, sigma, nb_simulations, K, type_payoff, frequence_mesure=1000,frequence_barriere="M", valeur_barriere=80, methode="Pseudo"):
    H = 0
    prix = []
    iterations = []
    for i in tqdm(range(1,nb_simulations+1)):
        S = black_scholes(S0, n, T, r, d, sigma, methode)
        H += payoff(S,K,type_payoff,T=T, frequence_barriere=frequence_barriere, valeur_barriere=valeur_barriere)
        if i % frequence_mesure == 0:
            esperance_payoff = H/i
            prix.append(np.exp(-r*T)*esperance_payoff)
            iterations.append(i)
    return prix, iterations

def halton(n, d=1):
    """
    Fonction uniforme entre 0 et 1
    """
    sampler = qmc.Halton(d, scramble=True)
    return sampler.random(n)

def halton_norm(n, d=1):
    sampler = qmc.Halton(d, scramble=True)
    x_halton = sampler.random(n)
    return stats.norm.ppf(x_halton)

# B = mouvement_brownien_standard(1000,1, methode="Quasi")
# plt.plot(B)
# plt.show()
# exit()


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

def autre_simul():
    Z = funcs[method](M)
    delta_St = nudt + volsdt*Z
    ST = S0*np.exp(delta_St)
    CT = np.maximum(0, ST - K)
    C0 = np.exp(-r*T)*np.sum(CT)/M
    prix = []
    iterations = []
    for i in tqdm(range(1,nb_simulations+1)):
        if i % frequence_mesure == 0:
            esperance_payoff = H/i
            prix.append(np.exp(-r*T)*esperance_payoff)
            iterations.append(i)
    return prix, iterations


result, iterations  = prix_instant_initial(S0, n, T, r, d, sigma,
                                           nb_simulations, K, "Call EU",
                                           frequence_mesure=50,
                                           frequence_barriere=frequence_barriere,
                                           valeur_barriere=H,
                                           methode="Pseudo")

print(result[-1])
# 1.2564532648137718 avec 1 million de simulations, Call Asiatique
# 3.441207795600795 avec 1 million de simu, Call EU
# 3.4409126762138484 avec 1 million de simu, Call down and out
plt.plot(iterations, result)
plt.axhline(y=3.441207795600795, color='r')
plt.savefig("Call_EU.png")
plt.show()