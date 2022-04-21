#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(start=-1.3, stop=2.5, num=64, dtype='float32').round(2)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    if cartesian_coordinates.ndim == 1:
        x = cartesian_coordinates[0]
        y = cartesian_coordinates[1]
        r = np.sqrt(x ** 2 + y ** 2)
        teta = np.arctan2(y, x)
        tableau = np.array([r, teta])
    else:
        tableau = np.zeros(cartesian_coordinates.shape, dtype='float16')
        for i in range(int(cartesian_coordinates.size/cartesian_coordinates.ndim)):
            x = cartesian_coordinates[i,0]
            y = cartesian_coordinates[i,1]
            r = np.sqrt(x ** 2 + y ** 2)
            teta = np.arctan2(y, x)
            tableau[i] = [r, teta]
    return tableau

def find_closest_index(values: np.ndarray, number: float): # je ne peux pas forcer le type de sortie car j'ai aussi besoin d'une liste
    if values.ndim == 1: # if dim==1
        return int(np.where(values==number)[0])
    else:
        return np.where(values == number)[0] # with the 0 u get only the first appearence
        #https://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array

#X = np.linspace(start=-1, stop=1, num=250) #demandé mais plus beau avec le 2ième
X = np.arange(-1,1,0.0001)
Y = (X**2)*(np.sin(1/(X**2))) + X
def draw_function(x,y):
    fig = plt.figure('Fonction', figsize=(10, 10))
    plt.plot(x, y)
    plt.title('Trigo_Function')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['GRAPH'])
    plt.show()

def integrale():
    X = np.arange(-4,4,0.01)
    #Y = np.exp(-1*(X**2))
    def f(x):
        return np.exp(-1*(x**2))

    Ih, err = quad(f, np.NINF, np.Inf) #Intégrale de -inf à +inf
    print("Integrale =", Ih, " Erreur =", err)

#En réalité on fait juste calculer les F(X) et on les associes sur le graphe avec F(X)
    def F(x): #FOnction qui va calculer tous les F(X) et les placer dans un tableau
        result = np.zeros_like(x) #Vecteur qui va contenir les F(X)
        for i, val in enumerate(x): #On parcours X et on calcule chaque image
            result[i] = quad(f, 0, val)[0]
        return result #     https://scicomp.stackexchange.com/questions/21869/plot-integral-function-with-scipy-and-matplotlib
    plt.plot(X, F(X))
    plt.show()

def approx_pi(nb_pt=100):
    # il me faut 1000 points différents / x^2+y^2<=1 x>=0 et y>=0
    points = np.zeros((nb_pt, 2), dtype='float16')
    liste1 = []
    liste2 = []
    compteur = 0
    nb_total_pt = 0
    while compteur < nb_pt:
        x = random.random()
        y = random.random()
        nb_total_pt +=1
        if ((x**2)+(y**2)<=1):
            liste1.append(x)
            liste2.append(y)
            compteur +=1
    points = np.array([liste1, liste2])
    print('L valeur approchée de PI est:', ((4 * compteur) + 1) / nb_total_pt)
    #Formule de PI: https://jpq.pagesperso-orange.fr/proba/montecarlo/index.htm#:~:text=Le%20calcul%20de%20%CF%80%20par,de%20disque%20de%20rayon%201.
    plt.plot(points[0], points[1],'ro')
    plt.show()




if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    a = linear_values()
    print(linear_values())
    print(coordinate_conversion(np.array([1, 0])))
    print(coordinate_conversion(np.array([[1,0], [0,1]])))
    print(find_closest_index(a, 2.5))
    print(find_closest_index(coordinate_conversion(np.array([[1,0], [0,1]])),1))
    #approx_pi(10000)
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a.shape)
    #draw_function(X,Y)
    integrale()
