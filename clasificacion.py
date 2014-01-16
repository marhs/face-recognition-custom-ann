import reconocimientoImagenes as rI

""" 
(El archivo principal es reconocimientoImagenes.py, este se usa para resolver
los problemas)

Clasificacion de imagenes

Primero, clasificar a una persona entre 5 posibles. Estro requiere una red
neuronal con 900 entradas (30x30) y 5 salidas, una por cada persona. 

Como conjunto de entrenamiento vamos a usar todas las imagenes menos las
siguientes: 1 (Maria), 8 (Irene), 16 (Pablo), 23 (Berta), 29 (Eva)

He usado una red con 1 capa oculta, y 15 nodos en esa capa. 
Factor aprendizaje: 0.1

Esta es la clasificacion que mejor ha funcionado, ya que, mientras que en las
otras tiene que fijarse en una parte de la persona (ojos, boca), en esta puede
fijarse en toda la foto, (pelo, boca, ojos, barbillas, cejas, barba) para
ayudar a clasificar. Por eso con solo 1000 iteraciones ya clasifica muy bien. 
"""

# Creamos el conjunto de entrenamiento
entrenamiento = rI.cargarImagenes(len(rI.entrenamiento))
print('[Reconocer la persona de la foto] Imagenes usadas: ',len(entrenamiento))
clasificacion = []
for n in entrenamiento:
    res = [0,0,0,0,0]
    res[n[1][0]-1] = 1
    n[1] = res

# Eliminamos los casos que vamos a clasificar, para que no se entrene con ellos

a = 1
for n in [1,8,16,23,29]:
    clasificacion.append(entrenamiento.pop(n-a))
    a+=1
print('     [Imagenes entrenamiento] :',len(entrenamiento))
print('     [Imagenes clasificacion] :',len(clasificacion))
# Creamos la red neuronal con la que vamos a trabajar. 
redCaras = rI.RedNeuronal([900,15,5])
#redCaras.retrop(entrenamiento,0.1,1000)
#rI.guardarRed(redCaras,'redCaras')
redCaras = rI.abrirRed('redCaras')
for n in clasificacion:
    print('    ',redCaras.salida(n[0]),n[1])


"""
Ahora intentamos saber si la persona en cuestion tiene los ojos abiertos o
cerrados. De nuevo, vamos a utilizar todo el conjunto de entrenamiento menos 3
fotos, para clasificarlas. Serán las fotos 12,22 y 34 (como cerrados), 5 y 23
como abiertos. 

Uno de los problemas que tiene a la hora de clasificar es que muchas veces hay
"ojos abiertos" que parecen ojos cerrados. Por ejemplo, cuando alguien está
mirando hacia abajo, tiene los ojos abiertos (y está clasificado como tal) pero
es muy fácil que se confunda con alguien que tiene los ojos cerrados. 

Por otra parte, determinadas personas de los conjuntos de entrenamiento tienden
a entornar los ojos cuando sonrien. 

"""

entrenamiento = rI.cargarImagenes(len(rI.entrenamiento))
clasificacion = []
print('[Ojos abiertos/cerrados: Retropropagación] Imagenes usadas =',len(entrenamiento))
for n in entrenamiento:
    res = [0,0]
    res[n[1][1]] = 1
    n[1] = res
a = 1
for n in [5,12,22,23,34]:
    clasificacion.append(entrenamiento.pop(n-a))
    a += 1
print('     [Imagenes entrenamiento] :',len(entrenamiento))
print('     [Imagenes clasificacion] :',len(clasificacion))
redOjos = rI.RedNeuronal([900,3,2])
#redOjos.retrop(entrenamiento,0.3,10000)
#rI.guardarRed(redOjos,'redOjos')
redOjos = rI.abrirRed('redOjos')
for n in clasificacion:
    print('    ',redOjos.salida(n[0]),n[1])

# Con regla Delta. 
# He usado como criterio de parada las 5000 iteraciones (unos 20 minutos de
# entrenamiento), y ha funcionado bastante bien, ya que con las 1000 usuales
# fallaba.

print('[Ojos abiertos/cerrados: Regla Delta] Imagenes usadas =',len(entrenamiento))
entrenamiento = rI.cargarImagenes(len(rI.entrenamiento))
clasificacion = []
for n in entrenamiento:
    res = [n[1][1]]
    n[1] = res
a = 1
for n in [5,12,22,23,34]:
    clasificacion.append(entrenamiento.pop(n-a))
    a += 1
print('     [Imagenes entrenamiento] :',len(entrenamiento))
print('     [Imagenes clasificacion] :',len(clasificacion))
redOjos = rI.RedNeuronal([900,1])
#a = redOjos.reglaDelta(entrenamiento,0.2,10000)
#rI.guardarRed(redOjos,'redOjosDelta')
redOjos = rI.abrirRed('redOjosDelta')
for n in clasificacion:
    print('    ',redOjos.salida(n[0]),n[1])


"""
Direccion de mirada
Clasificacion:  Arriba 13,39
                Centro  1,20
                Abajo  7

Aqui he usado una capa oculta de 4 unidades, aunque no he conseguido que
clasifique bien, ya que a veces, cuando debería estar mirando hacia abajo, en
realidad tiene los ojos cerrados. Hay gente que entorna mucho los ojos,
mientras que otros lo abren demasiado, gente que levanta la cabeza cuando mira
hacia arriba. 

Supongo que con un conjunto de entrenamiento mayor, se debería poder clasificar
de una manera más eficaz. 

"""

ent_dir = rI.cargarImagenes2(len(rI.entrenamiento2))
clasificacion = []

print('[Direccion de la mirada] Imagenes usadas = ',len(ent_dir))
for n in ent_dir:
    res = [0,0,0]
    res[n[1][2]] = 1
    n[1] = res

a=1
for n in [1,7,13,20,39]:
    clasificacion.append(ent_dir.pop(n-a))
    a += 1

print('     [Imagenes entrenamiento] :',len(ent_dir))
print('     [Imagenes clasificacion] :',len(clasificacion))

redDir = rI.RedNeuronal([900,4,3])
#redDir.retrop(ent_dir,0.1,10000)
#rI.guardarRed(redDir,'redDir')
redDir = rI.abrirRed('redDir')
print('[Clasificacion direccion mirada]')
for n in clasificacion:
    print('    ',redDir.salida(n[0]),n[1])

"""
Gestos con la boca. Retropropagacion
Imagenes para clasificacion: Serio:3 Risa:17 lengua:21 

En este caso, con 3000 iteraciones de retropropagacion, clasifica muy bien. 
"""
# Creamos el conjunto de entrenamiento
entrenamiento = rI.cargarImagenes(len(rI.entrenamiento))
print('[Reconocer el gesto de la persona] Imagenes usadas: ',len(entrenamiento))
clasificacion = []
for n in entrenamiento:
    res = [0,0,0]
    res[n[1][3]-1] = 1
    n[1] = res

# Eliminamos los casos que vamos a clasificar, para que no se entrene con ellos

a = 1
for n in [3,17,31]:
    clasificacion.append(entrenamiento.pop(n-a))
    a+=1
print('     [Imagenes entrenamiento] :',len(entrenamiento))
print('     [Imagenes clasificacion] :',len(clasificacion))
# Creamos la red neuronal con la que vamos a trabajar. 
redCaras = rI.RedNeuronal([900,15,3])
#redCaras.retrop(entrenamiento,0.2,3000)
#rI.guardarRed(redCaras,'redBocas')
redCaras = rI.abrirRed('redBocas')
for n in clasificacion:
    print('    ',redCaras.salida(n[0]),n[1])



