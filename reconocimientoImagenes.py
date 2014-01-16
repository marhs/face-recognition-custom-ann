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


# Implementacion de las funciones sigmoide y la derivada de sigmoide. 
def sigmoide(x):
# Funcion sigmoide f(x) = 1 / 1+e^-x
    # Ponemos estos límites para no tener que calcular con numeros muy grandes
    if(x>1000):
        return 1
    elif(x<-1000):
        return 0
    else:
        return 1/(1+math.exp(-x))

def sigmoideD(x):
    return sigmoide(x)*(1.0-sigmoide(x))
"""
Una red neuronal, a nivel de estructura de datos, es una lista con matrices.
Por una parte, (para una red con n capas, incluyendo capas ocultas, entrada
y salida) los pesos es una lista con n-1 matrices, describiendo cada una de 
ellas todos los pesos entre la capa i y j. 
Por otra parte, tenemos otras 2 listas (valores y valsigm) que almacenan
temporalmente los valores de cada capa y los valores de la funcion de
activacion. Obviamente, esto no necesita ser guardado en la red, pero ayuda
a mantener un control sobre el entrenamiento, o la clasificacion que se está
llevando a cabo. 

Por último, una lista auxiliar, que se pasa como parámetro de entrada
matrizTamanos: En esta lista, definimos el número de nodos de cada capa (e
indirectamente el número de capas. Es decir, que si creamos una red con la
lista [9,3,5,2], le estamos diciendo que queremos una red con 2 capas ocul-
tas, de 3 y 5 nodos respectivamente, 9 nodos en la capa de entrada y 2 en 
la capa de salida. 

Los pesos se general aleatoriamente entre [-1,1], pero podemos crear a mano
editando self.pesos. 

La implementación se ha hecho de tal manera que acepta cualquier número de
capas ocultas, así como con sus pesos. 

Respecto al criterio de parada, he usado un número determinado de iteraciones,
ya que no he conseguido encontrar ningún otro que me funcionara bien. 


"""

class RedNeuronal():
# Clase RedNeuronal para representar las redes neuronales. 

    def __init__(self, matrizTamanos):
        # Matriz - los tamaños de cada capa [entrada, co1,..,con, salida]
        # Es decir, la cantidad de unidades en cada capa, siendo la primera las
        # entradas y la ultima las salidas (por lo tanto el tamaño mínimo es 2)
        self.matrizTamanos = matrizTamanos
        
        # Matriz de valores. Cada elemento de esta lista es una lista con los
        # valores de cada capa. valores[0] = [x,x,x,x..] - entradas
        # En el algoritmo del libro esto es in
        self.valores = self.iniciaValores(matrizTamanos)         
        #  ""               ""              aj
        self.valsigm = self.iniciaValores(matrizTamanos)


        # Lista de las matrices con los pesos. La matriz[0] serán los pesos de
        # las entradas a la primera capa (o salida), la segunda de esta primera
        # capa a la siguiente, y así hasta llegar a la salida. Debe haber n+1
        # matrices, si hay n capas ocultas (ej: 1 capa oculta, 2 matrices)
        # 
        # En cada matriz m, m[0][1] sera el peso de ir del elemento 0 de la
        # capa 1 al elemento 1 de la capa 2. (por ejemplo)
        self.pesos = self.generaPesosAleatorios(self.matrizTamanos)
    
    
    # Funcion para inicializar las matrices (a 0) que despues se rellenaran
    def iniciaValores(self, matrizTamanos):
        res = []
        for n in matrizTamanos:
            subres = []
            for m in range(n):
                subres.append(0)
            res.append(subres)

        return res

 
    def generaMatrizAleatoria(self, n, m):
    # Genera una matriz con numeros aleatorios entre [-1,1] de n*m
        res = []
        for x in range(n):
            fila = []
            for y in range(m):
                fila.append((random()-0.5)*2)
            res.append(fila)

        return res

    def generaPesosAleatorios(self, matrizTamanos):
    # Generacion de pesos aleatorios para empezar a trabajar con la red. 
        res = []
        for x in range(len(matrizTamanos)):
            if x != 0:
                res.append(self.generaMatrizAleatoria(matrizTamanos[x-1],matrizTamanos[x]))
        return res

    def salida(self, caso):
    # Devuelve la salida de la red, dado una entrada (caso) y teniendo en
    # cuenta los pesos ya calculados
        valores = [caso]
        valsigm = [caso]

        for capa in range(len(self.pesos)):
            valores_capa = []
            valores_sigm = []
            for nodo in range(self.matrizTamanos[capa+1]):
                suma = 0
                for nodo_old in range(self.matrizTamanos[capa]):
                    suma += valsigm[capa][nodo_old]*self.pesos[capa][nodo_old][nodo]
                valores_capa.append(suma)
                valores_sigm.append(sigmoide(suma))
            valores.append(valores_capa)
            valsigm.append(valores_sigm)

        return valsigm[len(valsigm)-1]


    
    def reglaDelta(self, entrenamiento, factorAprendizaje, iteraciones):
        # Regla Delta para perceptrones simples. 

        # Comprobamos que sea un perceptrón. (no tenga capas ocultas) y que
        # solo tenga 1 salida. 
        if(len(self.matrizTamanos) != 2 or self.matrizTamanos[1] != 1):
            return False

        # Generamos los pesos aleatoriamente, aunque en la práctica ya están
        # generados. 
        # Definimos la condicion de terminacion

        # La condicion de finalizacion que usamos son iteraciones (las que
        # indiquemos)
        for n in range(iteraciones):
            for x,y in entrenamiento:
                ino = 0
                ws = []
                for entrada in range(len(x)):
                    ino += self.pesos[0][entrada][0] * x[entrada]
                    o = sigmoide(ino)
                for w in range(len(self.pesos[0])):
                    a = self.pesos[0][w][0] 
                    # Actualizacion de pesos. 
                    self.pesos[0][w][0]+=factorAprendizaje*(y[0]-o)*sigmoideD(ino)*x[w]
                    
        return self



    def retrop(self, entrenamiento, alpha,iteraciones):
    # Algoritmo de retropropagacion
    

        # Creamos un array para guardar los deltas de cada nodo. En un
        # principio estaran a 0, pero luego se irán actualizando, cuando
        # hagamos la siguiente parte. 
        # Errores [capa][nodo]
        errores = []
        for capa in range(len(self.matrizTamanos)):
            suberrores = []
            for nodo in range(self.matrizTamanos[capa]):
                suberrores.append(0)
            errores.append(suberrores)

        for it in range(iteraciones):
            # Muestra las iteraciones que lleva, para algoritmos largos
            # (algunos me han tardado bastante en entrenar)
            #if it%100 == 0:
            #    print((100/iteraciones)*it, '%')

            for x,y in entrenamiento:
                # Metemos las entradas en sus respectivos lugares. 
                self.valores[0] = x
                # Se supone que el valsigm van los valores sigmoide(valor),
                # pero en la primera capa no hace falta, así que lo rellenamos
                # con el entrenamiento
                self.valsigm[0] = x 
                # Propagamos hasta la salida. 
                # Recorremos cada capa y hacemos los cálculos
                for capa in range(len(self.pesos)):
                    valoresCapa = []
                    valoresSigma = []
                    for nodo in range(self.matrizTamanos[capa+1]):
                        sumEntradasPeso = 0
                        for old in range(len(self.valores[capa])):
                            # Hacemos la suma*pesos de toda la red
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
                        # Actualizamos los errores para el resto de nodos. 
                        errores[n][nodo] = sigmoideD(self.valores[n][nodo]) * aux

                        # Actualizamos los pesos. 
                        for nodosig in range(len(errores[n+1])):
                            self.pesos[n][nodo][nodosig] += alpha*self.valsigm[n][nodo]*errores[n+1][nodosig]

        return self
        
# Lectura de las imágenes del conjunto de entrenamiento. 
# He decidido no leer cada pixel de la imagen en un int de [0-255] (que es como
# lo hacia en un principio, si no como un float entre [0-1], ya que a la hora
# de operar, hace que las sumas no salgan tan altas. 

def leerImagen(nombre):
    # Devuelve un array 30x30
    file = open(nombre, 'rb')
    for n in range(13):
        file.read(1)
    res = []
    for n in range(30):
        for m in range(30):
            res.append(float(struct.unpack('c', file.read(1))[0][0]/255))

    return res



""" 
Funciones auxiliares para guardar/abrir redes neuronales desde un archivo
"""
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

# Generación de conjuntos de entrenamiento aleatorios para pruebas. 
# Los conjuntos de entrenamiento son matrices con todos los ejemplos de
# entrenamiento (x,y). Cada ejemplo es una tupla (x,y) con x e y arrays de
# datos. 

# No se usa en el trabajo, pero me ha servido para ir probando las redes, así
# como los diferentes tipos de algoritmos (cuando quería usar un conjunto de
# entrenamiento muy grande)

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
Ambos conjuntos de entrenamiento
La clave es el numero de la foto, y la lista indica las propiedades de esta
[x1,x2,x3,x4]
x1: Id persona (1:Maria, 2:Pablo, 3:Pablo, 4:Berta, 5:Eva)
x2: 1: Ojos Abiertos, 0: Ojos Cerrados
x3: Direccion ojos 0: arriba, 1: centro, 2: abajo
x4: Boca 0: Serio, 1: Sonriendo, 2:Lengua fuera

Despues, para cada una de las clasificaciones, usaremos estos datos para crear
el conjunto de entrenamiento

Para la obtencion de ambos conjuntos de entrenamiento, se han tomado fotos con
una cámara digital (RICOH R10), en formato jpeg. Usando la libreria netpgm se
han transformado a formato pgm, y posteriormente centradas, retocadas y
redimensionadas a 30x30 con Adobe Photoshop CS5. 

"""
entrenamiento = {
        1  : [1,1,1,1],
        2  : [1,1,1,0],
        3  : [1,1,2,0],
        4  : [1,1,0,0],
        5  : [1,1,1,1],
        6  : [1,1,1,2],
        7  : [1,0,2,0],
        8  : [2,1,1,0],
        9  : [2,1,1,1],
        10 : [2,1,2,1],
        11 : [2,0,2,0],
        12 : [2,1,2,0],
        13 : [2,1,0,0],
        14 : [2,1,1,2],
        15 : [2,1,2,1],
        16 : [3,0,1,0],
        17 : [3,1,1,1],
        18 : [3,1,0,0],
        19 : [3,0,2,0],
        20 : [3,1,1,2],
        21 : [3,0,2,0],
        22 : [3,0,2,0],
        23 : [4,1,1,0],
        24 : [4,0,1,1],
        25 : [4,0,1,0],
        26 : [4,1,0,1],
        27 : [4,0,2,1],
        28 : [4,1,2,2],
        29 : [5,1,1,0],
        30 : [5,1,1,1],
        31 : [5,1,1,2],
        32 : [5,0,2,1],
        33 : [5,1,0,1],
        34 : [5,1,2,1]}

entrenamiento2 = {
        1  : [1,0,1,0],
        2  : [1,0,1,1],
        3  : [1,1,1,1],
        4  : [1,0,1,0],
        5  : [1,0,1,0],
        6  : [1,0,0,0],
        7  : [1,1,2,0],
        8  : [1,0,1,2],
        9  : [1,0,1,2],
        10 : [1,1,2,0],
        11 : [1,1,2,0],
        12 : [1,1,2,0],
        13 : [1,0,0,0],
        14 : [1,1,2,0],
        15 : [1,0,1,0],
        16 : [1,0,1,1],
        17 : [2,0,1,0],
        18 : [2,0,1,1],
        19 : [2,0,1,1],
        20 : [2,0,1,1],
        21 : [2,0,1,2],
        22 : [2,0,0,0],
        23 : [2,1,2,0],
        24 : [2,0,1,0],
        25 : [2,0,1,0],
        26 : [2,0,1,0],
        27 : [1,0,1,0],
        28 : [1,0,1,0],
        29 : [1,1,2,0],
        30 : [1,1,2,0],
        31 : [1,1,2,0],
        32 : [1,1,2,0],
        33 : [1,1,2,2],
        34 : [1,1,2,2],
        35 : [1,1,2,2],
        36 : [1,0,1,2],
        37 : [1,0,1,2],
        38 : [1,0,1,2],
        39 : [1,0,0,2],
        40 : [1,0,0,0],
        41 : [1,0,0,0],
        42 : [1,0,0,0],
        43 : [1,1,2,0],
        44 : [1,1,2,0],
        45 : [1,1,2,0],
        46 : [1,1,2,0],
        47 : [1,0,1,0],
        48 : [1,1,1,1],
        49 : [1,0,1,0],
        50 : [1,0,1,0]}

# Estas dos funciones cargan las imagenes de ambos conjuntos de entrenamiento
def cargarImagenes(num):
    res = []
    for n in range(1,num+1):
        a = leerImagen('img/'+str(n)+'.pgm')
        b = entrenamiento[n]
        res.append([a,b])
    return res

def cargarImagenes2(num):
    res = []
    for n in range(1,num+1):
        a = leerImagen('img/con2/'+str(n)+'.pgm')
        b = entrenamiento2[n]
        res.append([a,b])
    return res

"""
Test que clasifica correctamente con reglaDelta (pero dentro de los conjuntos
de entrenamiento, no con imágenes nuevas
"""

"""
print('[Test regla Delta 1, datos pequeños]')
red = RedNeuronal([3,1])
ent = [([0,0,1],[0]),([0,1,1],[0]),([1,0,1],[1])]
print(' [Entrenamiento] : ',ent)
red.pesos = [[[0.5],[0.5],[0.5]]]
print(' [Pesos]         : ',red.pesos)
red.reglaDelta(ent,0.1,100000)
clasifica = [1,1,0]
print(' ',clasifica,' clasifica como ',red.salida([1,0,1]))
print(' [Pesos actuales]: ',red.pesos)

"""
"""
Test que clasifica correctamente con retropropagacion dentro del conjunto de
entrenamiento. 
"""

"""
print('[Test Retropropagacion, datos pequeños]')
red = RedNeuronal([2,2,2])
ent = [ ([0,0],[0,0]),
        ([0,1],[0,1]),
        ([1,0],[0,1]),
        ([1,1],[1,1])]
print(' [Entrenamiento] : ',ent)
print(' [Pesos]         : ',red.pesos)
red.retrop(ent,0.1,10000)
clasifica = [0,1]
print(' ',clasifica,' es clasificado como ', red.salida(clasifica))
print(' [Pesos actuales]: ',red.pesos)

"""
