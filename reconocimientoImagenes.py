# Trabajo IA - Tecnologías Informáticas
# Marco Herrero Serna <hielo.helado@gmail.com>

# Reconocimento de imágenes mediante redes neuronales artificiales. 

# Uso math para la constante e y random para generar pesos aleatoriamente. 
import math
from random import random
# Pickle se usa para el guardado de objetos 
import pickle
# Struct lo usamos para leer imágenes binarias
import struct
# Constante e, sacada de math. 
e = math.e

def sigmoide(x):
# Funcion sigmoide f(x) = 1 / 1+e^-x
    return 1/(1+math.pow(e,-x))

def sigmoideD(x):

    return sigmoide(x)*(1.0-sigmoide(x))

class RedNeuronal():
# Clase RedNeuronal para representar las redes neuronales. 

    def __init__(self, matrizTamanos):
        # Matriz - los tamaños de cada capa [entrada, co1,..,con, salida]
        # Es decir, la cantidad de unidades en cada capa, siendo la primera las
        # entradas y la ultima las salidas (por lo tanto el tamaño mínimo es 2)
        self.matrizTamanos = matrizTamanos
        
        # Matriz de valores. Cada elemento de esta lista es una lista con los
        # valores de cada capa. valores[0] = [x,x,x,x..] - entradas
        self.valores = self.iniciaValores(matrizTamanos)  # En el algoritmo del libro esto es in
        self.valsigm = self.iniciaValores(matrizTamanos)  #  ""                               aj

        
        # Lista de las matrices con los pesos. La matriz[0] serán los pesos de
        # las entradas a la primera capa (o salida), la segunda de esta primera
        # capa a la siguiente, y así hasta llegar a la salida. Debe haber n+1
        # matrices, si hay n capas ocultas (ej: 1 capa oculta, 2 matrices)
        # 
        # En cada matriz m, m[0][1] sera el peso de ir del elemento 0 de la
        # capa 1 al elemento 1 de la capa 2. (por ejemplo)
        self.pesos = self.generaPesosAleatorios(self.matrizTamanos)
    
    
    def iniciaValores(self, matrizTamanos):
        res = []
        for n in matrizTamanos:
            subres = []
            for m in range(n):
                subres.append(0)
            res.append(subres)

        return res

    def generaMatrizAleatoria(self, n, m):
    # Genera una matriz con numeros aleatorios entre [0,1] de n*m
        res = []
        for x in range(n):
            fila = []
            for y in range(m):
                fila.append(0.5)
            res.append(fila)

        return res

    def generaPesosAleatorios(self, matrizTamanos):
    # Generacion de pesos aleatorios para empezar a trabajar con la red. 
        res = []
        for x in range(len(matrizTamanos)):
            if x != 0:
                res.append(self.generaMatrizAleatoria(matrizTamanos[x-1],matrizTamanos[x]))
        return res

    def salida(self):
    # TODO Convertir salida en el calculo del nodo N (oculto o final) dados sus
    # nodos anteriores
    # Define la salida i del perceptrón. Hace la operacion g(sum(a(i)*w(i))
        res = 0
        for entrada in range(len(self.datos)):
            res += self.datos[entrada]*self.pesos[entrada]
        return res
        

    def reglaDelta(self, entrenamiento, factorAprendizaje):
    # TODO No funciona para valores muy altos del conjunto de entrenamiento
    # (por ejemplo, 255). Habría que reducir los valores proporcionalmente o
    # algo así.
    # Regla Delta para perceptrones simples. 

        # Comprobamos que sea un perceptrón. 
        if(len(self.matrizTamanos) != 2 or self.matrizTamanos[1] != 1):
            return False

        # Generamos los pesos aleatoriamente, aunque en la práctica ya están
        # generados. 
        # Definimos la condicion de terminacion

        # TODO while not condicionTerminacion:
        for x,y in entrenamiento:
            ino = 0
            ws = []
            for entrada in range(len(x)):
                ino += self.pesos[0][entrada][0] * x[entrada]
                o = sigmoide(ino)
                
            for w in range(len(self.pesos[0])):
                a = self.pesos[0][w][0] 
                self.pesos[0][w][0] = self.pesos[0][w][0] + factorAprendizaje*(y[0]-o) * sigmoideD(ino) * x[w]
        return self.pesos

    def retrop(self, entrenamiento, alpha):
    # Algoritmo de retropropagacion

        # Errores [capa][nodo]
        errores = []
        for capa in range(len(self.matrizTamanos)):
            suberrores = []
            for nodo in range(self.matrizTamanos[capa]):
                suberrores.append(0)
            errores.append(suberrores)
        # Definimos la condicion de terminacion

        # while not condicionTerminacion:

        for x,y in entrenamiento:
            # Metemos las entradas en sus respectivos lugares. 
            print(self.pesos)    
            self.valores[0] = x
            self.valsigm[0] = x # TODO Rellenarlo de ceros, vago. 
            # Propagamos hasta la salida. 
            # Recorremos cada capa
            for capa in range(len(self.pesos)):
                valoresCapa = []
                valoresSigma = []
                for nodo in range(self.matrizTamanos[capa+1]):
                    sumEntradasPeso = 0
                    for old in range(len(self.valores[capa])):
                        sumEntradasPeso += self.valsigm[capa][old]*self.pesos[capa][old][nodo]
                    valoresCapa.append(sumEntradasPeso)
                    valoresSigma.append(sigmoide(sumEntradasPeso))
                self.valores[capa+1] = valoresCapa
                self.valsigm[capa+1] = valoresSigma
            # Definimos los errores para la capa de salida
            err = []
            for nodo in range(len(self.valores[len(self.valores)-1])):
                delta = sigmoideD(self.valores[len(self.valores)-1][nodo])*(y[nodo]-self.valsigm[len(self.valsigm)-1][nodo])
                err.append(delta)
            errores[len(self.matrizTamanos)-1] = err
        
            # Recorrer las capas inversamente, empezando por la anterior a
            # salida. 
            for n in reversed(range(len(self.valores)-1)):
                # Ahora, en cada capa, recorremos sus nodos
                for nodo in range(len(self.valores[n])):
                    aux = 0
                    for nodosig in range(len(errores[n+1])):
                        aux += errores[n+1][nodosig]*self.pesos[n][nodo][nodosig]
                    errores[n][nodo] = sigmoideD(self.valores[n][nodo]) * aux

                    for nodosig in range(len(errores[n+1])):
                        self.pesos[n][nodo][nodosig] += alpha*self.valsigm[n][nodo]*errores[n+1][nodosig]

        return self
        
# Lectura de las imágenes del conjunto de entrenamiento. 

def leerImagen(nombre):
    # Devuelve un array 30x30
    file = open(nombre, 'rb')
    for n in range(13):
        file.read(1)
    res = []
    for n in range(30):
        for m in range(30):
            res.append(int(struct.unpack('c', file.read(1))[0][0]))

    return res



# Parte 3: Aprendizaje para reconocer caras


# Parte auxiliar:

def writeRed(red):
# Funcion para guardar una red neuronal a un archivo. 
    with open('red_data.pk', 'wb') as output:
        pickle.dump(red, output, pickle.HIGHEST_PROTOCOL)

def readRed():
    with open('red_data.pk', 'rb') as input:
        res = pickle.load(input)
        return res

def guardarRed(red, nombre):
# Guardamos la red neuronal a un archivo. Limpiamos los valores y dejamos
# solo los pesos, que es lo que nos importa. 
    red.valores = red.iniciaValores(red.matrizTamanos)  
    red.valsigm = red.iniciaValores(red.matrizTamanos)

    with open(nombre, 'wb') as output:
        pickle.dump(red, output, pickle.HIGHEST_PROTOCOL)

    return nombre

def abrirRed(nombre):
    with open(nombre, 'rb') as inp:
        red = pickle.load(inp)
    return red

# Zona de testeo
# TODO Tener un test (o varios) preparados para cada parte, y así probarlos
# delante del profesor. 

# Generación de conjuntos de entrenamiento aleatorios para pruebas. 
# Los conjuntos de entrenamiento son matrices con todos los ejemplos de
# entrenamiento (x,y). Cada ejemplo es una tupla (x,y) con x e y arrays de
# datos. 

def generacionEntrenamiento(entradas, salidas, casos):
    d = []
    for caso in range(casos):
        x = []
        for entrada in range(entradas):
            x.append(random())
        y = []
        for salida in range(salidas):
            y.append(random())
        d.append((x,y))
    return d

"""
a = leerImagen('foo.pgm')
print(a)
b = leerImagen('bar.pgm')
print(b)
red = RedNeuronal([900,1])
ent = [(a,[0]),(b,[1])]
print(red.pesos)
print(red.reglaDelta(ent,0.1))

print('--')
red = RedNeuronal([2,1])
print('Pesos red: ',red.pesos)
print(red.reglaDelta(ent,0.1))
"""

"""
# Prueba de entrenamiento de la regla Delta
ent = [([34,255,2],[0]),([111,32,44],[1])]
rn = RedNeuronal([3,1])
rn.pesos = [[[0.5],[0.5],[0.5]]]
print(rn.pesos)
print('Retro, primer intento')
print(rn.reglaDelta(ent,0.1))
print(rn.pesos[0])
"""
# Prueba de entrenamiento de algoritmo de retropropagación

ent = [([1,2],[0,1])]
rn = RedNeuronal([2,2,2])
rn.retrop(ent,0.1)
print(rn.pesos)
#print(RedNeuronal([[0]]).generaPesosAleatorios([2,1,3,2]))

