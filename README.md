Reconocimiento de caras y expresiones mediante redes neuronales articifiales
======================
Face and expression recognition using artificial neural networks. 
[Spanish version]
 
Reconocimiento de caras y expresiones mediante redes neuronales artificiales y un conjunto de entrenamiento. 
Este código es la solución al trabajo de la asignatura Inteligencia Artificial de la ETSII - Universidad de Sevilla, 2013. Dentro de los archivos del proyecto se encuentra el enunciado de este trabajo. Está presentado y puntuado, por lo que no lo mantendré durante mucho más tiempo, a menos que sea muy necesario o se me ocurra retomar el proyecto. Se que el código tiene algunos fallos y bugs, y que debería estar estructurado de otra manera, así que si tengo tiempo lo arreglaré. 

Para ejecutar el clasificador de imagenes, hay que poner el conjunto de entrenamiento en una carpeta llamada "img". Dentro de reconocimientoImagenes.py hay un dictionary de python con las puntuaciones introducidas a mano sobre cada foto del conjunto de entrenamiento. Esto está documenado dentro del archivo. 

Para finalizar, ejecutar clasificador.py para las pruebas. 

Nota: Un error del trabajo es la ausencia de entradas de sesgo W0 en los nodos de la red, así que clasifica peor de lo que debería y a veces produce resultado no esperado. Ten esto en cuenta si vas a utilizar el código. 



