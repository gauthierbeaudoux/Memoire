import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from time import time
from math import sqrt, log, exp
from scipy.stats import norm

def price_call_euro(S0, r, d, sigma, T, t, K):
    d1 = (log(S0/K) + (r-d+sigma**2/2)*(T-t))/(sigma*sqrt(T-t))
    d2 = d1 - sigma*sqrt(T-t)
    return S0*exp(-d*(T-t))*norm.cdf(d1)-K*exp(-r*(T-t))*norm.cdf(d2)

# print(price_call_euro(100, 0.05, 0.03, 0.1, 1, 0.5, 103))

def mouvement_brownien(n, T, x0, r, sigma):
    h = T/n
    X = np.zeros(n+1)
    X[0] = x0
    for i in range(n):
        X[i+1] = X[i] + np.random.normal(r*h, sigma*np.sqrt(h))
    return X

def mouvement_brownien_standard(n, T):
    return mouvement_brownien(n, T, 0, 0, 1)

def black_scholes(S0, n, T, r, d, sigma):
    W = mouvement_brownien_standard(n, T)
    h = T/n
    return [S0*np.exp((r - d - sigma**2/2)*(t*h) + sigma*W[t]) for t in range(n+1)]

# for _ in range(10):
#     plt.plot(black_scholes(100, 1000, 1, 0.05, 0.03, 0.1))
# plt.show()

def payoff(W, K):
    return max(W[-1]-K, 0)

def price(S0, r, d, sigma, T, K, n, N):
    payoffs = []
    for _ in tqdm(range(N)):
        payoffs.append(payoff(black_scholes(S0, n, T, r, d, sigma), K))
    actu_payoffs = [np.exp(-r*T)*p for p in payoffs]
    mu = sum(actu_payoffs)/N
    sigma = np.sqrt(sum((p-mu)**2 for p in actu_payoffs)/N)
    return mu,sigma 

"""
est = 3.4336826904838884

Nb_sim = 10000

E = []
P100 = [price(100, 0.05, 0.03, 0.1, 1, 103, 1, 100)[0]]
P = []

#print(price(100, 0.05, 0.03, 0.1, 1, 103, 1, 1000000))

t = 0

for _ in tqdm(range(Nb_sim)):
    ta = time()
    P100.append(price(100, 0.05, 0.03, 0.1, 1, 103, 1, 100)[0])
    t += time()-ta

t /= Nb_sim

T = [t*k for k in range(1, Nb_sim+1)]

for k in tqdm(range(Nb_sim)):
    P.append(1/(k+1)*sum(P100[i] for i in range(k+1)))

#est = np.mean(P100)

#for k in range(len(N)//2):
#    print(sqrt(1/2*(P)))

for k in tqdm(range(1,Nb_sim+1)):
    e = 0
    for i in range(Nb_sim//k):
        e += abs((1/k)*sum(P100[k*i + j] for j in range(k))-est)/est
    E.append(e/(Nb_sim//k))

#df  = pd.DataFrame({'Temps': T, 'Valeur': P, 'Erreur': E})
#df.to_csv("C:/Users/tomdu/OneDrive/Bureau/Centrale/Master FQ/Mémoire/Graphes/call_down_in.csv", index=False)

print(np.mean(P100))

plt.plot(T, E, color="blue", alpha=0.5)
plt.show()

"""

def hist_payoff(S0, r, d, sigma, T, K, n, N):
    H = []
    for _ in tqdm(range(N)):
        H.append(payoff(black_scholes(S0, n, T, r, d, sigma), K))
    fig, ax = plt.subplots()
    ax.hist(H, bins = 100, density=True)
    plt.show()
    return H

def mouvement_brownien_fast(n, T, x0, r, sigma):
    h = T/n
    X = np.zeros(n)
    X[0] = x0
    for i in range(n-1):
        X[i+1] = X[i] + np.random.normal(r*h, sigma*np.sqrt(h))
    return X

def mouvement_brownien_standard_fast(n, T):
    return mouvement_brownien_fast(n, T, 0, 0, 1)

def black_scholes_fast(S0, n, T, r, d, sigma):
    W = mouvement_brownien_standard_fast(n, T)
    h = T/n
    return [S0*np.exp((r - d - sigma**2/2)*(t*h) + sigma*W[t]) for t in range(n)]

def price_fast(S0, r, d, sigma, T, K, n, N):
    new_K = 0
    for _ in tqdm(range(N)):
        W = black_scholes_fast(S0, n, T, r, d, sigma)
        new_K += K - sum([W[t] for t in range(1,n)])/n
    new_K /= N
    return price_call_euro(S0/n, r, d, sigma, T, T-T/n, new_K)

print(price(100, 0.05, 0.03, 0.1, 1, 103, 2, 100_000))
print(price_fast(100, 0.05, 0.03, 0.1, 1, 103, 2, 100_000))