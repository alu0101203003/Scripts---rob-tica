#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Rob�tica Computacional 
# Grado en Ingenier�a Inform�tica (Cuarto)
# Pr�ctica 5:
#     Simulaci�n de robots m�viles holon�micos y no holon�micos.

import sys
from math import *
from robot import robot
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# ******************************************************************************
# Declaraci�n de funciones

def distancia(a,b):
  # Distancia entre dos puntos (admite poses)
  return np.linalg.norm(np.subtract(a[:2],b[:2]))

def angulo_rel(pose,p):
  # Diferencia angular entre una pose y un punto objetivo 'p'
  w = atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]
  while w >  pi: w -= 2*pi
  while w < -pi: w += 2*pi
  return w

def mostrar(objetivos,ideal,trayectoria):
  # Mostrar objetivos y trayectoria:
  plt.ion() # modo interactivo
  # Fijar los bordes del gr�fico
  objT   = np.array(objetivos).T.tolist()
  trayT  = np.array(trayectoria).T.tolist()
  ideT   = np.array(ideal).T.tolist()
  bordes = [min(trayT[0]+objT[0]+ideT[0]),max(trayT[0]+objT[0]+ideT[0]),
            min(trayT[1]+objT[1]+ideT[1]),max(trayT[1]+objT[1]+ideT[1])]
  centro = [(bordes[0]+bordes[1])/2.,(bordes[2]+bordes[3])/2.]
  radio  = max(bordes[1]-bordes[0],bordes[3]-bordes[2])*.75
  plt.xlim(centro[0]-radio,centro[0]+radio)
  plt.ylim(centro[1]-radio,centro[1]+radio)
  # Representar objetivos y trayectoria
  idealT = np.array(ideal).T.tolist()
  plt.plot(idealT[0],idealT[1],'-g')
  plt.plot(trayectoria[0][0],trayectoria[0][1],'or')
  r = radio * .1
  for p in trayectoria:
    plt.plot([p[0],p[0]+r*cos(p[2])],[p[1],p[1]+r*sin(p[2])],'-r')
    #plt.plot(p[0],p[1],'or')
  objT   = np.array(objetivos).T.tolist()
  plt.plot(objT[0],objT[1],'-.o')
  plt.show()
  input()
  plt.clf()

def localizacion(balizas, real, ideal, centro, radio, mostrar=0):
  # Buscar la localizaci�n m�s probable del robot, a partir de su sistema
  # sensorial, dentro de una regi�n cuadrada de centro "centro" y lado "2*radio" (diámetro).

  # Tamaño de cada pixel de la imagen
  FACTOR_LOCALIZACION = 0.05
  lado_size = int(round((2 * radio) / FACTOR_LOCALIZACION))
  imagen = []
  
  # Se rellena toda la matriz con 0
  for y in range(lado_size):
   imagen.append([])
   for x in range(lado_size):
     imagen[y].append(0)
  
  # Se calculan las distancias que van desde el robot real a las balizas
  distancias_real = real.sense(balizas)

  # Se asigna infinito como valor por defecto para el mejor peso (será sustituido al calcular)
  mejor_peso = float('inf')
  mejor_x = 0
  mejor_y = 0

  # Se recorrera la imagen para asignarle segun la posición del robot real y ideal los pesos a cada pixel y posteriormente se obtendrá el punto con menor peso
  # Se inicializan las coordenadas en la esquina de la imagen y se recorren (en los ejes y,x)
  y = centro[1] - radio 
  for i in range(lado_size): 
    x = centro[0] - radio
    for j in range(lado_size):
      # Se establece la pose del robot ideal
      ideal.set(x, y, ideal.orientation)
      # Se calcula el peso (error entre el robot ideal y las medidas reales)
      peso = ideal.measurement_prob(distancias_real, balizas)
      imagen[i][j] = peso

      # Se actualiza el peso buscando, en cada caso, el mejor
      if (peso < mejor_peso):
        mejor_peso = peso
        mejor_x = x
        mejor_y = y
    # Se avanza en x e y un pixel (factor)
      x += FACTOR_LOCALIZACION 
    y += FACTOR_LOCALIZACION 

  # Se establece la pose del robot ideal en función del mejor peso obtenido
  ideal.set(mejor_x, mejor_y, ideal.orientation)
  # Se muestran los resultados: componentes de la pose y peso
  print ("('Mejor: ', [", ideal.pose()[0], ", ", ideal.pose()[1], "], ' ', ", mejor_peso)

  if mostrar:
    plt.ion() # modo interactivo
    plt.xlim(centro[0]-radio,centro[0]+radio)
    plt.ylim(centro[1]-radio,centro[1]+radio)
    imagen.reverse()
    plt.imshow(imagen,extent=[centro[0]-radio,centro[0]+radio,\
                              centro[1]-radio,centro[1]+radio])
    balT = np.array(balizas).T.tolist();
    plt.plot(balT[0],balT[1],'or',ms=10)
    plt.plot(ideal.x,ideal.y,'D',c='#ff00ff',ms=10,mew=2)
    plt.plot(real.x, real.y, 'D',c='#00ff00',ms=10,mew=2)
    plt.show()
    input()
    plt.clf()

# ******************************************************************************

# Definici�n del robot:
P_INICIAL = [0.,4.,0.] # Pose inicial (posici�n y orientacion)
V_LINEAL  = .7         # Velocidad lineal    (m/s)
V_ANGULAR = 140.       # Velocidad angular   (�/s)
FPS       = 10.        # Resoluci�n temporal (fps)

HOLONOMICO = 1
GIROPARADO = 0
LONGITUD   = .2

# Definici�n de trayectorias:
trayectorias = [
    [[1,3]],
    [[0,2],[4,2]],
    [[2,4],[4,0],[0,0]],
    [[2,4],[2,0],[0,2],[4,2]],
    [[2+2*sin(.8*pi*i),2+2*cos(.8*pi*i)] for i in range(5)]
    ]

# Definici�n de los puntos objetivo:
if len(sys.argv)<2 or int(sys.argv[1])<0 or int(sys.argv[1])>=len(trayectorias):
  sys.exit(sys.argv[0]+" <�ndice entre 0 y "+str(len(trayectorias)-1)+">")
objetivos = trayectorias[int(sys.argv[1])]

# Definici�n de constantes:
EPSILON = .1                # Umbral de distancia
V = V_LINEAL/FPS            # Metros por fotograma
W = V_ANGULAR*pi/(180*FPS)  # Radianes por fotograma
MAXIMO_PESO = 0.3

ideal = robot()
ideal.set_noise(0,0,.1)   # Ruido lineal / radial / de sensado
ideal.set(*P_INICIAL)     # operador 'splat'

real = robot()
real.set_noise(.01,.01,.1)  # Ruido lineal / radial / de sensado
real.set(*P_INICIAL)

random.seed(0)
trayectoria_ideal = [ideal.pose()]
trayectoria_real = [real.pose()]

tiempo  = 0.
espacio = 0.
#random.seed(0)

# Se localiza inicialmente el robot ideal
localizacion(objetivos, real, ideal, [2, 2], 3, 1)

random.seed(datetime.now())
for punto in objetivos:
  while distancia(trayectoria_ideal[-1],punto) > EPSILON and len(trayectoria_ideal) <= 1000:
    pose = ideal.pose()

    # Los movimientos no superarán los límites establecidos
    w = angulo_rel(pose,punto)
    if w > W:  w =  W
    if w < -W: w = -W
    v = distancia(pose,punto)
    if (v > V): v = V
    if (v < 0): v = 0

    if HOLONOMICO:
      if GIROPARADO and abs(w) > .01:
        v = 0
      ideal.move(w,v)
      real.move(w,v)
    else:
      ideal.move_triciclo(w,v,LONGITUD)
      real.move_triciclo(w,v,LONGITUD)
    trayectoria_ideal.append(ideal.pose())
    trayectoria_real.append(real.pose())

    real_measurements = real.sense(objetivos)
    # Comparamos el peso del robot ideal con las medidas reales
    weight = ideal.measurement_prob(real_measurements, objetivos)
    
    # Cuando el peso supere el umbral indicado, se corrige la pose del robot ideal
    if (weight > MAXIMO_PESO):
      localizacion(objetivos, real, ideal, ideal.pose(), 1)
    
    espacio += v
    tiempo  += 1

if len(trayectoria_ideal) > 1000:
  print ("<!> Trayectoria muy larga - puede que no se haya alcanzado la posici�n final.")
print ("Recorrido: "+str(round(espacio,3))+"m / "+str(tiempo/FPS)+"s")
print ("Distancia real al objetivo: "+\
    str(round(distancia(trayectoria_real[-1],objetivos[-1]),3))+"m")
mostrar(objetivos,trayectoria_ideal,trayectoria_real)  # Representaci�n gr�fica

