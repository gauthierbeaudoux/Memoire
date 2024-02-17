import random
import numpy as np

# Integrale de 0 à 1 de exp(x)

# Analytiquement vaut e-1=1.7183
print("Objectif : 1.7183")

nb_simulations = 10_000

somme_esp = 0
liste_tirage_1 = []
for i in range(nb_simulations):
    tirage_a = random.uniform(0,1)
    tirage_b = random.uniform(0,1)
    valeur = (np.exp(tirage_a) + np.exp(tirage_b))/2
    liste_tirage_1.append(valeur)
    
print(f"Espérance méthode 1 : {np.mean(liste_tirage_1):.4f}")
print(f"Variance 1 : {np.var(liste_tirage_1):.5f}")

somme_esp = 0
liste_tirage_2 = []
for i in range(nb_simulations):
    tirage_a = random.uniform(0,1)
    valeur = (np.exp(tirage_a) + np.exp(1-tirage_a))/2
    liste_tirage_2.append(valeur)
    
print(f"Espérance méthode 2 : {np.mean(liste_tirage_2):.4f}")
print(f"Variance 2: {np.var(liste_tirage_2):.5f}")


