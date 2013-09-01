# Trabajo IA - Tecnologías Informáticas
# Marco Herrero Serna <hielo.helado@gmail.com>

# Reconocimento de imágenes mediante redes neuronales artificiales. 

# TODO General
#   * Buscar una forma de guardar redes neuronales a archivo
#       - En este caso, tendríamos que guardar un objeto de una clase ya
#         creado, con sus matrices y todo. 
#       - También necesitamos una forma de recuperar redes desde archivo. 
#   * Escribir una funcion para, dado unas entradas (y los pesos actuales de la
#     red neuronal) devuelva una/s salida/s, para así probar si la estimacion
#     de los algoritmos es buena con los casos particulares. 

# Uso math para la constante e y random para generar pesos aleatoriamente. 
import math
from random import random

import pickle

# Constante e, sacada de math. 
e = math.e

def sigmoide(x):
# Funcion sigmoide f(x) = 1 / 1+e^-x
    return 1/(1+math.pow(e,-x))

def sigmoideD(x):

    return sigmoide(x)*(1.0-sigmoide(x))

class Perceptron():
# Clase perceptrón.

    def __init__(self, numEntradas):
    # De los perceptrones, así como de las redes neuronales solo sabemos su
    # estructura, de tal manera que al no haber capas ocultas solo tenemos en
    # cuenta el número de entradas y de salidas. 

        self.numEntradas = numEntradas
        self.entradas = []
        
        # Cada matriz dentro de esta será el valor de los nodos de las capas
        # ocultas. Tengo que buscar una forma de definirla dado el constructor.
        self.capasOcultas = [] 

        # Los pesos es una matriz donde las filas representan las salidas y
        # las columnas representan las entradas, por lo tanto el peso Wi,j
        # estará en la fila i, columna j
        # Es decir pesos[salida][entrada]
        # TODO Propagar el numero de capas a generaPesos. 
        self.pesos = self.generaPesosAleatorios(numEntradas) # TODO

    def generaPesosAleatorios(self, m):
    # Generacion de pesos aleatorios para empezar a trabajar con la red. 
        a = []
        for x in range(m):
            a.append(random())
        return a

    def salida(self):
    # TODO Convertir salida en el calculo del nodo N (oculto o final) dados sus
    # nodos anteriores
    # Define la salida i del perceptrón. Hace la operacion g(sum(a(i)*w(i))
        res = 0
        for entrada in range(len(self.entradas)):
            res += self.entradas[entrada]*self.pesos[entrada]
        return res
        

    def reglaDelta(self, entrenamiento, factorAprendizaje):
    # Regla Delta para perceptrones simples. 
    # TODO Si voy a hacer una red completa en la clase perceptron, tengo que
    # hacer que la regla delta solo se pueda aplicar si NO HAY capas ocultas,
    # no solo haciendo que de fallo. 
    
        # Generamos los pesos aleatoriamente, aunque en la práctica ya están
        # generados. 
        self.pesos = self.generaPesosAleatorios(self.numEntradas)
        print("Pesos iniciales: ", self.pesos) 
        # Definimos la condicion de terminacion

        # while not condicionTerminacion:
        for x,y in entrenamiento:
            ino = 0
            ws = []
            for entrada in range(len(x)):
                ino += self.pesos[entrada] * x[entrada]
                o = sigmoide(ino)
                
                for w in range(len(self.pesos)):
                    self.pesos[w] = self.pesos[w] + factorAprendizaje*(y-o) * sigmoideD(ino) * x[w]
            print("[Entrada] - ",x," - y - ", y, "  + ",self.pesos)
                
        return self.pesos
        
# Parte 2: Implementacion de algoritmos de aprendizaje de redes neuronales. 
#   Regla delta para entrenamiento del perceptron simple. 


#   Algoritmo de retropropagación


#   Algoritmo de retropropagación con momentum. 

# Parte 3: Aprendizaje para reconocer caras


# Parte auxiliar:

def writeRed(red):
# Funcion para guardar una red neuronal a un archivo. 
    with open('red_data.pk', 'wb') as output:
        pickle.dump(red, output, pickle.HIGHEST_PROTOCOL)

def readRed():
    with open('red_data.pk', 'rb') as input:
        res = pickle.load(input)
        print(res)


# Zona de testeo
# TODO Tener un test (o varios) preparados para cada parte, y así probarlos
# delante del profesor. 

# Generación de conjuntos de entrenamiento para pruebas. 
# Los conjuntos de entrenamiento son matrices con todos los ejemplos de
# entrenamiento (x,y). Cada ejemplo es una tupla (x,y) con x e y arrays de
# datos. 

def generacionEntrenamiento(entradas, salidas, casos):
    d = []
    for caso in range(casos):
        x = []
        for entrada in range(entradas):
            x.append(random())
        if salidas > 1:
            y = []
            for salida in range(salidas):
                y.append(random())
        else:
            y = random()
        d.append((x,y))
    return d
"""
ent = generacionEntrenamiento(3,1,2)
# Tests perceptrón
print('Generacion de Entrenamiento - 3 in, 1 out, 2 cases: ')
print(ent)
perceptron = Perceptron(3)
perceptron.entradas = [1,30,-10]
print(perceptron.reglaDelta(ent, 0.9))
print(perceptron.salida())
"""
arp = 'Prueba de escritura'
writeRed(arp)
readRed()
