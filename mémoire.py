import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from time import time

def temps(f, *args, L, N):
    T = []
    for k in tqdm(L):
        t = 0
        for _ in range(N):
            T0 = time()
            f(*args, k)
            t += time()-T0
        T.append(t/N)
    return T



def mouvement_brownien(n, T, x0, r, sigma):
    h = T/n
    X = np.zeros(n+1)
    X[0] = x0
    for i in range(n):
        X[i+1] = X[i] + np.random.normal(r*h, sigma*np.sqrt(h))
    return X
"""
temps = []

for k in tqdm(range(1,501)):
    t = 0
    for _ in range(100):
        T = time()
        mouvement_brownien(k, 1, 0, 0, 1)
        t += time()-T
    temps.append(t/100)
plt.plot(temps)
plt.show()
"""
def mouvement_brownien_standard(n, T):
    return mouvement_brownien(n, T, 0, 0, 1)

def anim_mouvement_brownien_standard(n, T):
    plt.ion()
    while True:
        plt.plot([k*T/n for k in range(n+1)], mouvement_brownien_standard(n, T))
        plt.xlabel(r'Temps $t$')
        plt.ylabel(r'Valeur de $W_t$')
        plt.show()
        plt.draw()
        plt.pause(0.5)

def enregistrer_mouvement_brownien_standard(n, T, N, filename):
    times = [k*T/n for k in range(n+1)]
    simulations = {'Time': times}
    for i in range(1, N+1):
        simulations[f'Simulation_{i}'] = mouvement_brownien_standard(n, T)
    df = pd.DataFrame(simulations)
    df.to_csv("C:/Users/tomdu/OneDrive/Bureau/Centrale/Master FQ/Mémoire/"+filename, index=False)

#enregistrer_mouvement_brownien_standard(1000, 1, 30, 'simulations_mouvement_brownien_standard.csv')

def black_scholes(S0, n, T, r, d, sigma):
    W = mouvement_brownien_standard(n, T)
    h = T/n
    return [S0*np.exp((r - d - sigma**2/2)*(t*h) + sigma*W[t]) for t in range(n+1)]

def anim_black_scholes(S0, n, T, r, d, sigma):
    plt.ion()
    while True:
        plt.plot(black_scholes(S0, n, T, r, d, sigma))
        plt.draw()
        plt.pause(0.5)

def enregistrer_black_scholes(S0, n, T, r, d, sigma, N, filename):
    times = [k*T/n for k in range(n+1)]
    simulations = {'Time': times}
    for i in range(1, N+1):
        simulations[f'Simulation_{i}'] = black_scholes(S0, n, T, r, d, sigma)
    df = pd.DataFrame(simulations)
    df.to_csv("C:/Users/tomdu/OneDrive/Bureau/Centrale/Master FQ/Mémoire/"+filename, index=False)

#enregistrer_black_scholes(100, 1000, 1, 0.5, 0.3, 0.1, 30, 'simulations_black_scholes.csv')

def payoff(W, K):
    return max(W[-1]-K, 0)

#anim_mouvement_brownien_standard(1000, 1)
#anim_black_scholes(100, 1000, 1, 0.05, 0.03, 0.1)

def quantile(p,S0, r, d, sigma, T, K, n, N):
    Q = []
    M = 1000000//N
    for _ in tqdm(range(M)):
        payoffs = []
        for _ in range(N):
            payoffs.append(payoff(black_scholes(S0, n, T, r, d, sigma), K))
        payoffs.sort()
        Q.append(payoffs[int(N*p)-1])
    return np.mean(Q)

Q = []
"""
for k in range(1,11):
    Q.append(quantile(0.95, 100, 0.05, 0.03, 0.1, 1, 103, 1, 1000*k))

    
plt.plot([1000*k for k in range(1,11)], Q)
plt.show()
"""
def price(S0, r, d, sigma, T, K, n, N):
    payoffs = []
    for _ in tqdm(range(N)):
        payoffs.append(payoff(black_scholes(S0, n, T, r, d, sigma), K))
    actu_payoffs = [np.exp(-r*T)*p for p in payoffs]
    mu = sum(actu_payoffs)/N
    sigma = np.sqrt(sum((p-mu)**2 for p in actu_payoffs)/N)
    return mu,sigma 

#mu, sigma = price(100, 0.5, 0.2, 0.1, 1, 103, 1000, 50000)
#print(mu, sigma)

MU = []

M = 100

P = 1000

#plt.plot([n for n in range(M,M*P)], [3.44 + 1.96*5.71/np.sqrt(n) for n in range(M,M*P)])
#plt.plot([n for n in range(M,M*P)], [3.44 - 1.96*5.71/np.sqrt(n) for n in range(M,M*P)])
#plt.show()

"""
for N in [M*k for k in range(1,11)]:
    mu, sigma = price(100, 0.05, 0.03, 0.1, 1, 103, 1000, N)
    MU.append(mu)


plt.plot([M*k for k in range(1,11)], MU)
plt.show()
"""
T = []

for k in range(1,11):
    t = time()
    print(price(100, 0.05, 0.03, 0.1, 1, 103, 1, 10000*k))
    T.append(time()-t)

plt.plot([10000*k for k in range(1,11)], T)
plt.show()

def hist_payoff(S0, r, d, sigma, T, K, n, N):
    H = []
    for _ in tqdm(range(N)):
        H.append(payoff(black_scholes(S0, n, T, r, d, sigma), K))
    fig, ax = plt.subplots()
    ax.hist(H, bins = 100, density=True)
    plt.show()
    return H
"""
for k in range(1,11):
    hist_payoff(100, 0.5, 0.3, 0.1, 1, 103, 1, 1000*k)
"""
    
#hist_payoff(100, 0.5, 0.3, 0.1, 1, 103, 1000, 1000)

def enregistrer_payoff(S0, r, d, sigma, T, K, n, N, filename):
    H = []
    simulation = {'Payoff': H}
    for _ in tqdm(range(N)):
        H.append(payoff(black_scholes(S0, n, T, r, d, sigma), K))
    df = pd.DataFrame(simulation)
    df.to_csv("C:/Users/tomdu/OneDrive/Bureau/Centrale/Master FQ/Mémoire/"+filename, index=False)

#hist_payoff(100, 0.5, 0.3, 0.1, 1, 103, 100, 50000)
#enregistrer_payoff(100, 0.5, 0.3, 0.1, 1, 103, 1000, 50000, 'simulations_payoff.csv')