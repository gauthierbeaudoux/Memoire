import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import qmc
import scipy.stats as stats
import timeit
import time
import pandas as pd



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
        case "Put down and out":
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
                    return max(K-S[-1], 0)
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




# B = mouvement_brownien_standard(1000,1, methode="Quasi")
# plt.plot(B)
# plt.show()
# exit()





# temps_sans_numba = timeit.timeit(stmt="autre_simul(nb_simulations, S0, K, d, T)",
#                                  globals=globals(),
#                                  number=100)
# print(f"Temps sans numba : {round(temps_sans_numba, 5)}")


# result, iterations = autre_simul(nb_simulations, S0, K, d, T)

def autre_simul(nb_simulations, S0, K, d, T):
    #precompute constants
    dt = T
    nudt = (r - d - 0.5*sigma**2)*dt
    volsdt = sigma*np.sqrt(dt)
    liste_nb_simu = np.linspace(10,nb_simulations, 3)
    result = []
    for x in liste_nb_simu:
        x = int(x)
        # Z = np.random.normal(loc=0, scale=1, size=x)
        Z = halton_norm(x)
        delta_St = nudt + volsdt*Z
        ST = S0*np.exp(delta_St)
        CT = np.maximum(0, ST - K)
        C0 = np.exp(-r*T)*np.sum(CT)/x
        result.append(C0)
        
    return result, liste_nb_simu

def autre_simul_numba(nb_simulations, S0, K, d, T):
    #precompute constants
    dt = T
    nudt = (r - d - 0.5*sigma**2)*dt
    volsdt = sigma*np.sqrt(dt)
    liste_nb_simu = np.linspace(10,nb_simulations, 3)
    result = []
    for x in liste_nb_simu:
        x = int(x)
        # Z = np.random.normal(loc=0, scale=1, size=x)
        Z = halton_norm(x)
        delta_St = nudt + volsdt*Z
        ST = S0*np.exp(delta_St)
        CT = np.maximum(0, ST - K)
        C0 = np.exp(-r*T)*np.sum(CT)/x
        result.append(C0)
        
    return result, liste_nb_simu

# Partie Pseudo vs Quasi

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

def sobol(m, d=1):
    sampler = qmc.Sobol(d, scramble=True)
    return sampler.random_base2(m)

def sobol_norm(m, d=1):
    sampler = qmc.Sobol(d, scramble=True)
    x_sobol = sampler.random_base2(m)
    return stats.norm.ppf(x_sobol)

def graph_pseudo_random() -> None:
    n = 500
    x1 = np.random.uniform(0,1,int(n))
    x2 = np.random.uniform(0,1,int(n))

    plt.scatter(x1,x2, marker='d')
    plt.show()
    

def graph_quasi_random() -> None:
    from scipy.stats import qmc

    def halton(n, d=1):
        sampler = qmc.Halton(d, scramble=True)
        return sampler.random(n)

    x = halton(n=500, d=2).T
    plt.scatter(x[0],x[1], marker='d')
    plt.show()


def pricing_blackScholes_formula(r, S, K, T, sigma, type="c"):
        "Calculate BS price of call/put"
        d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        try:
            if type == "c":
                price = S*stats.norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*stats.norm.cdf(d2, 0, 1)
            elif type == "p":
                price = K*np.exp(-r*T)*stats.norm.cdf(-d2, 0, 1) - S*stats.norm.cdf(-d1, 0, 1)
            return price
        except:
            print("Please confirm option type, either 'c' for Call or 'p' for Put!")




def comparaison_cv_quasi_vs_pseudo():
    # Define variables
    r = 0.01
    S0 = 30
    K = 32
    T = 1
    vol = 0.30
    bs = pricing_blackScholes_formula(r, S0, K, T, vol, type="c")
    print('Black Scholes Price', round(bs,3))

    results = {
            # 'Pseudo: add_12_uni': [],
            # 'Pseudo: box_muller': [],
            # 'Pseudo: polar_rejection:': [],
            # 'Pseudo: inv_transform': [],
            'Pseudo : Numpy': [],
            'Quasi : Halton': [],
            'Quasi : Sobol': []
           }

    funcs = {
            # 'Pseudo: add_12_uni': add_12_uni,
            # 'Pseudo: box_muller': box_muller,
            # 'Pseudo: polar_rejection:': polar_rejection,
            # 'Pseudo: inv_transform': inverse_norm,
            'Pseudo : Numpy': np.random.standard_normal,
            'Quasi : Halton': halton_norm,
            'Quasi : Sobol': sobol_norm
            }

    numbers = np.linspace(0,4000,21)[1:]
    # N = 10000

    #precompute constants
    dt = T
    nudt = (r - 0.5*vol**2)*dt
    volsdt = vol*np.sqrt(dt)

    # Monte Carlo Method
    for M in numbers:
        M = int(M)
        for method in results:
            if method == 'Quasi : Sobol':
                continue
            else:
                Z = funcs[method](M)
            delta_St = nudt + volsdt*Z
            ST = S0*np.exp(delta_St)
            CT = np.maximum(0, ST - K)
            C0 = np.exp(-r*T)*np.sum(CT)/M

            results[method].append(C0 - bs)

    sobol_rng = np.arange(7,13)
    for M in sobol_rng:
        M = int(M)

        Z = funcs['Quasi : Sobol'](M)
        delta_St = nudt + volsdt*Z
        ST = S0*np.exp(delta_St)
        CT = np.maximum(0, ST - K)
        C0 = np.exp(-r*T)*np.sum(CT)/(2**M)

        results['Quasi : Sobol'].append(C0 - bs)


    sigma = np.sqrt( np.sum( (np.exp(-r*T)*CT - C0)**2) / (M-1) )
    SE = sigma/np.sqrt(M)
    
    # Plot
    plt.figure(figsize=(8,5))
    for method in results:
        if method == 'Quasi : Sobol':
            plt.plot(2**sobol_rng,results[method],label=method,color='k',marker='+')
        else:
            plt.plot(numbers,results[method],label=method,marker='+')
    
    df = pd.DataFrame({
    "Nb_iterations": numbers,
    "Pseudo : Numpy": results["Pseudo : Numpy"],
    "Quasi : Halton": results["Quasi : Halton"],
    })
    df.to_csv("Comparaison_finale.xlsx")
        
    df_sobol = pd.DataFrame({
    "Nb_iterations_sobol": 2**sobol_rng,
    "Quasi : Sobol": results["Quasi : Sobol"],
    })
    df_sobol.to_csv("Comparaison_finale_sobol.csv")
    



    plt.legend()
    plt.title('Convergence de Monte-Carlo \n Nombres Pseudo vs Quasi-aléatoire')
    plt.ylabel('Erreur de pricing relative')
    plt.xlabel('Nombre de simulations')
    plt.plot()
    plt.show()

def black_scholes_with_barrier(S, K, T, r, sigma, H):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    d3 = (np.log(S / H) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d4 = d3 - sigma * np.sqrt(T)
    d5 = (np.log(H**2 / (S * K)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d6 = d5 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    knock_in_barrier = S * (H / S)**(2 * (r / (sigma**2) - 0.5)) * norm.cdf(d3) - K * np.exp(-r * T) * (H / S)**(2 * (r / (sigma**2))) * norm.cdf(d4)
    knock_out_barrier = S * (H / S)**(2 * (r / (sigma**2) - 0.5)) * norm.cdf(d5) - K * np.exp(-r * T) * (H / S)**(2 * (r / (sigma**2))) * norm.cdf(d6)

    return call_price - (knock_out_barrier - knock_in_barrier)


S0 = 100
sigma = 0.1
r = 0.05
d = 0.03
K = 103
T = 1
n = 12
nb_simulations = 10**5
H = 95
frequence_barriere = "M"
type_payoff = "Put down and out"


'''
result, iterations  = prix_instant_initial(S0, n, T, r, d, sigma,
                                           nb_simulations, K, type_payoff,
                                           frequence_mesure=50,
                                           frequence_barriere=frequence_barriere,
                                           valeur_barriere=H,
                                           methode="Pseudo")


# print(f"{pricing_blackScholes_formula(r, S0, K, T, sigma, type='c') = }")
# print(f"{black_scholes_with_barrier(S0, K, T, r, sigma, H) = }")
print(f"Résultat = {result[-1]}")
df = pd.DataFrame({
    "Nb_iterations": iterations,
    "Put_down_and_out": result,
    })
df.to_csv("Put_down_and_out.csv")
# 1.2564532648137718 avec 1 million de simulations, Call Asiatique
# 3.441207795600795 avec 1 million de simu, Call EU
# 3.4409126762138484 avec 1 million de simu, Call down and out
plt.plot(iterations, result, label=type_payoff)
plt.legend()
# plt.axhline(y=3.441207795600795, color='r')
# plt.savefig(f"{type_payoff}_{nb_simulations}_{H}.png")
plt.show()
'''

comparaison_cv_quasi_vs_pseudo()


