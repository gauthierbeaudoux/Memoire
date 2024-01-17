from math import tan, pi, sqrt, exp
from random import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def inv():
    return 1/(1+random()**2)

def monte_carlo(N):
    s = 0
    for _ in range(N):
        s += inv()
    return s/N

def gauss_approx():
    x = random()
    y = (1/0.77-sqrt(1/0.77**2-2*(0.46/0.77)*x))/(0.46/0.77)
    return (1/(1+y**2))/((1-0.46*y)/0.77)

def reduc_var(N):
    s = 0
    for _ in range(N):
        s += gauss_approx()
    return s/N

def gauss_approx2():
    x = random()
    return (1/(1+x**2)-(1-0.46*x))

def reduc_var2(N):
    s = 0
    for _ in range(N):
        s += gauss_approx2()
    return s/N+0.77

def gauss_approx3():
    x = random()
    return (1/(1+x**2)+1/(1+(1-x)**2))/2

def reduc_var3(N):
    s = 0
    for _ in range(N):
        s += gauss_approx3()
    return s/N

#print(4*monte_carlo(10000000))
#print(4*reduc_var(10000000))
#print(4*reduc_var2(10000000))
#print(4*reduc_var3(10000000))

V = []

N = 1000

for _ in tqdm(range(10000)):
    V.append(monte_carlo(N) )

P = [4*(sum(V[:i+1])/(i+1)) for i in range(len(V))]
E = [abs(P[i] - pi)/pi for i in range(len(P))]

VA = []

for _ in tqdm(range(10000)):
    VA.append(reduc_var(N) )

PA = [4*(sum(VA[:i+1])/(i+1)) for i in range(len(VA))]
EA = [abs(PA[i] - pi)/pi for i in range(len(PA))]

VA2 = []

for _ in tqdm(range(10000)):
    VA2.append(reduc_var2(N))

PA2 = [4*(sum(VA2[:i+1])/(i+1)) for i in range(len(VA2))]
EA2 = [abs(PA2[i] - pi)/pi for i in range(len(PA2))]

VA3 = []

for _ in tqdm(range(10000)):
    VA3.append(reduc_var3(N))

PA3 = [4*(sum(VA3[:i+1])/(i+1)) for i in range(len(VA3))]
EA3 = [abs(PA3[i] - pi)/pi for i in range(len(PA3))]

plt.plot([N*k for k in range(1,10001)], E, color='red', label='Monte Carlo', alpha=0.5)
plt.plot([N*k for k in range(1,10001)], EA, color='blue', label='Reduced Variance', alpha=0.5)
plt.plot([N*k for k in range(1,10001)], EA2, color='green', label='Reduced Variance 2', alpha=0.5)
plt.plot([N*k for k in range(1,10001)], EA3, color='orange', label='Reduced Variance 3', alpha=0.5)
plt.legend()
plt.show()