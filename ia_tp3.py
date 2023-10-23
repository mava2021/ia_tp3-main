import numpy as np  # Librería para el manejo de matrices
import random  # Librería para generar números aleatorios
from PIL import Image  # Libreria para el manejo de imágenes
import os  # Librería para el manejo de funciones del S.O.
import re  # Librería para el manejo de expresiones regulares.

# Función convertir Matriz a Vector
def matriz_vector(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i, j]
            c += 1
    return tmp1


# Función para hacer matriz de pesos de una imagen
def crear_MatrizP(x):
    if len(x.shape) != 1:
        print("La entrada no es un vector")
        return
    else:
        w = np.zeros([len(x), len(x)])
        for i in range(len(x)):
            for j in range(i, len(x)):
                if i == j:
                    w[i, j] = 0
                else:
                    w[i, j] = x[i]*x[j]
                    w[j, i] = w[i, j]
    return w

# Función para transformar una imágen a un array Numpy
def imagen_array(archivo, dimension):
    pilIN = Image.open(archivo).convert(mode="L")
    pilIN = pilIN.resize(dimension)
    imgArray = np.asarray(pilIN, dtype=np.uint8)
    x = np.zeros(imgArray.shape, dtype=np.float64)
    x[imgArray > 0] = 1
    x[x == 0] = -1
    return x

# Función para convertir un array Numpy a una imágen
def array_imagen(data, archivoSalida=None):
    # convierte los datos a 1 o -1 en la matriz
    y = np.zeros(data.shape, dtype=np.uint8)
    y[data == 1] = 255  # Color blanco
    y[data == -1] = 0  # Color negro
    img = Image.fromarray(y, mode="L")
    if archivoSalida is not None:
        img.save(archivoSalida)
    return img


# Actualizar
def actualiza(peso, y_vec, theta=0.5, pasos=100):
    for s in range(pasos):
        m = len(y_vec)
        i = random.randint(0, m-1)
        u = np.dot(peso[i][:], y_vec) - theta

        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1
        # pasos
    return y_vec


# Comienzo del entrenamiento
def entrenar(entrenador_archivos, testeo_archivos, theta=0.5, pasos=1000, dimension=(10, 10), ruta_actual=None):

    # Leer la imagen y convertirla a un array Numpy
    print("Importando imagenes y creando matriz de pesos....")

    # Cantidad de archivos
    cant_archivos = 0
    for ruta in entrenador_archivos:
        print(ruta)
        x = imagen_array(archivo=ruta, dimension=dimension)
        x_vec = matriz_vector(x)
        # Imprimir vector entrenador
        print(x_vec, "\n")
        print("Tamaño: ", len(x_vec))
        if cant_archivos == 0:
            peso = crear_MatrizP(x_vec)
            cant_archivos = 1
        else:
            peso_tmp = crear_MatrizP(x_vec)
            peso = peso + peso_tmp
            cant_archivos += 1

    print("Matriz de pesos finalizada!")

    # Importar informacion de testeo
    counter = 0
    for ruta in testeo_archivos:
        y = imagen_array(archivo=ruta, dimension=dimension)
        oshape = y.shape
        y_img = array_imagen(y)
        y_img.show()
        print("Informacion de test importada")
        y_vec = matriz_vector(y)
        print("Actualizando...")
        y_vec_despues = actualiza(peso=peso, y_vec=y_vec, theta=theta, pasos=pasos)
        y_vec_despues = y_vec_despues.reshape(oshape)
        if ruta_actual is not None:
            archivoSalida = ruta_actual+"/salida_"+str(counter)+".png"
            img_despues = array_imagen(y_vec_despues, archivoSalida=archivoSalida)
            img_despues.show()
        else:
            img_despues = array_imagen(y_vec_despues, archivoSalida=None)
            img_despues.show()
        counter += 1


# Main
# Lista de archivos para entrenar
ruta_actual = os.getcwd()
entrenador_rutas = []
ruta = ruta_actual+"/entrenador_archivos/"
for i in os.listdir(ruta):
    # wildcard para cualquier nombre de archivo siempre con extension png
    if re.match(r'[0-9a-zA-Z-]*.png', i):
        entrenador_rutas.append(ruta+i)

# Lista de archivos para analizar / testear
testeo_rutas = []
ruta = ruta_actual+"/testeo_archivos/"
for i in os.listdir(ruta):
    # wildcard para cualquier nombre de archivo siempre con extension png
    if re.match(r'[0-9a-zA-Z-_]*.png', i):
        testeo_rutas.append(ruta+i)

# Iniciar entrenamiento Hopfield
# entrenador_archivos: contiene todos los archivos .png que se encuentran en la carpeta ./entrenador_archivos
# entrenador_rutas: contiene todos los archuvis .png que se encuentran en la carpeta ./testeo_archivos
# dimension: dimensión en pixeles.
entrenar(entrenador_archivos=entrenador_rutas, testeo_archivos=testeo_rutas,
         theta=0.5, pasos=1000, dimension=(10, 10), ruta_actual=ruta_actual)
