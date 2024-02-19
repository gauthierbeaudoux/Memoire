import random
import numpy as np
import timeit

"""
Source : https://www.youtube.com/watch?v=dihn4djMaqw
"""

# Integrale de 0 à 1 de exp(x)

f_exp = lambda x: np.exp(x)

print("Fonction exponentielle :\n")

# Analytiquement vaut e-1=1.7183
print("Objectif : 1.7183")

nb_simulations = 10_000

# Définition de la fonction pour estimer l'espérance et la variance
def simu_esp_var(nb_simulations, f, type="Classique"):
    liste_tirage = []
    # calcul_esp = 0
    # calcul_var = 0
    for _ in range(nb_simulations):
        match type:
            case "Classique":
                tirage_a = random.uniform(0,1)
                tirage_b = random.uniform(0,1)
                valeur = (f(tirage_a) + f(tirage_b))/2
                liste_tirage.append(valeur)
            case "Antithetic":
                tirage = random.uniform(0,1)
                valeur = (f(tirage) + f(1-tirage))/2
                liste_tirage.append(valeur)
            case "Control":
                tirage = random.uniform(0,1)
                # Pour Y, on prend la loi uniforme
                # On sait que E(Y) = 1/2 et V(Y) = 1/12
                liste_tirage.append(tirage)
            
        # calcul_esp += valeur
        # calcul_var += valeur**2
    
    if type == "Control":
        liste_x = [np.exp(x) for x in liste_tirage]
        esp_X = np.mean(liste_x)
        esp_Y = 1/2
        cov = np.mean([(np.exp(x)-esp_X)*(x-esp_Y) for x in liste_tirage])
        c = -cov/(1/12)
        esp_Z = esp_X + c*(np.mean(liste_tirage)-esp_Y)
        var_Z = np.var([liste_x[i] + c*(liste_tirage[i]-esp_Y) for i in range(len(liste_tirage))])
        return esp_Z, var_Z



    # return calcul_esp/nb_simulations, calcul_var/nb_simulations - (calcul_esp/nb_simulations)**2
    return np.mean(liste_tirage), np.var(liste_tirage)



# 1ère estimation : 2 tirages, puis moyenne classique

esperance1, variance1 = simu_esp_var(nb_simulations, f_exp)
print(f"Espérance méthode 1 : {esperance1:.4f}")
print(f"Variance 1 : {variance1:.5f}")


"""
2e estimation : 1 seul tirage avec moyenne grâce à 1-tirage
Méthode Antithetic
"""

esperance2, variance2 = simu_esp_var(nb_simulations, f_exp, type="Antithetic")  
print(f"Espérance méthode 2 : {esperance2:.4f}")
print(f"Variance 2: {variance2:.5f}")

'''
--> Meilleure estimation avec 2 fois moins de tirages et une variance 
beaucoup plus faible

----------------------------------------------------------------------

!!! Attention, ne fonctionne qu'avec des fontions monotones !!!


Exemple avec l'intégrale de 0 à 1 de (x-1/2)^2
'''
print("\n\n")
print("Fonction qui à x associe (x-1/2)^2 :\n")

f_non_mono = lambda x: (x-1/2)**2

# 1ère estimation : 2 tirages, puis moyenne classique

esperance1, variance1 = simu_esp_var(nb_simulations, f_non_mono)
print(f"Espérance méthode 1 : {esperance1:.4f}")
print(f"Variance 1 : {variance1:.5f}")

# 2e estimation : 1 seul tirage avec moyenne grâce à 1-tirage

esperance2, variance2 = simu_esp_var(nb_simulations, f_non_mono, type="Antithetic")  
print(f"Espérance méthode 2 : {esperance2:.4f}")
print(f"Variance 2: {variance2:.5f}")

'''
On remarque que la variance est plus élevée dans le 2e cas !


3e estimation : Control variates


Idée :
On calcule E(X) ) l'aide de Z et Y tel que :
Z = X + c(Y-mu) avec E(Y) = mu
On a donc bien E(Z) = E(X)

Il faut trouver c pour que V(Z) soit minimale.

En calculant, on trouve : c_min = -Cov(X,Y)/V(Y)


!!! IMPORTANT : E(Y) doit être connu avant la simulation !!!

'''

esperance3, variance3 = simu_esp_var(nb_simulations, f_exp, type="Control")
print(f"Espérance méthode 3 : {esperance3:.4f}")
print(f"Variance 3 : {variance3:.5f}")


'''
Conclusion pour l'exponentielle
'''

nombre_temps = 100

esperance1, variance1 = simu_esp_var(nb_simulations, f_exp)
temps1 = min(timeit.repeat(stmt='simu_esp_var(nb_simulations, f_exp)',
                      globals=globals(), number=nombre_temps))

esperance2, variance2 = simu_esp_var(nb_simulations, f_exp, type="Antithetic")  
temps2 = min(timeit.repeat(stmt='simu_esp_var(nb_simulations, f_exp, type="Antithetic")',
                      globals=globals(), number=nombre_temps))

esperance3, variance3 = simu_esp_var(nb_simulations, f_exp, type="Control")
temps3 = min(timeit.repeat(stmt='simu_esp_var(nb_simulations, f_exp, type="Control")',
                      globals=globals(), number=nombre_temps))

print("\nComparaison globale :")
print(f"Classique  : {esperance1:.4f}, {variance1:.5f}, {temps1:.5f}")
print(f"Antithesis : {esperance2:.4f}, {variance2:.5f}, {temps2:.5f}")
print(f"Control    : {esperance3:.4f}, {variance3:.5f}, {temps3:.5f}")
