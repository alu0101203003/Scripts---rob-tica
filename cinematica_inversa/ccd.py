#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - Curso 2014/2015
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs

# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O,final=0):
  # Muestra los orígenes de coordenadas para cada articulación
  print('Origenes de coordenadas:')
  for i in range(len(O)):
    print('(O'+str(i)+')0\t= '+str([round(j,3) for j in O[i]]))
  if final:
    print('E.Final = '+str([round(j,3) for j in final]))

def muestra_robot(O,obj):
  # Muestra el robot graficamente
  plt.figure(1)
  plt.xlim(-L,L)
  plt.ylim(-L,L)
  T = [np.array(o).T.tolist() for o in O]
  for i in range(len(T)):
    plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i/float(len(T)),1,1))
  plt.plot(obj[0], obj[1], '*')
  plt.show()
  input()
  plt.clf()

def matriz_T(d,th,a,al):
  # Calcula la matriz T (ángulos de entrada en grados)
  
  return [[cos(th), -sin(th)*cos(al),  sin(th)*sin(al), a*cos(th)]
         ,[sin(th),  cos(th)*cos(al), -sin(al)*cos(th), a*sin(th)]
         ,[      0,          sin(al),          cos(al),         d]
         ,[      0,                0,                0,         1]
         ]

def cin_dir(th,a):
  #Sea 'th' el vector de thetas
  #Sea 'a'  el vector de longitudes
  T = np.identity(4)
  o = [[0,0]]
  for i in range(len(th)):
    T = np.dot(T,matriz_T(0,th[i],a[i],0))
    tmp=np.dot(T,[0,0,0,1])
    o.append([tmp[0],tmp[1]])
  return o


# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# método para obtener el ángulo que forman 2 vectores

def obtener_angulo(a, b, c):
  vector_ca = np.subtract(a, c)
  vector_cb = np.subtract(b, c)

  alfa_1 = atan2(vector_ca[1], vector_ca[0])
  alfa_2 = atan2(vector_cb[1], vector_cb[0])

  tita = alfa_1 - alfa_2
  return tita

# valores articulares arbitrarios para la cinemática directa inicial
th=[0.,0.,0.]
a =[5.,5.,5.]
tipo_articulaciones = [0, 0, 0]
L = sum(a)
EPSILON = .01

plt.ion()

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
  sys.exit("python " + sys.argv[0] + " x y")
objetivo=[float(i) for i in sys.argv[1:]]

O=list(range(len(th)+1))
# cálculo de la cinemática directa
O[0]=cin_dir(th,a) 
print("- Posicion inicial:")
muestra_origenes(O[0])

# establecemos los límites a 45 grados
limites_max = [np.radians(45), np.radians(45), np.radians(45)]
limites_min = [np.radians(-45), np.radians(-45), np.radians(45)]

dist = float("inf")
prev = 0.
iteracion = 1
while (dist > EPSILON and abs(prev-dist) > EPSILON/100.):
  prev = dist
  # Para cada combinación de articulaciones:
  for i in list(range(len(th))):

    # cálculo de la cinemática inversa:
    art_actual = len(th) - 1 - i # articulación en cada iteración (índice de posiciones)
    art_final = O[i][-1] # punto final del robot (con el que se quiere lograr la convergencia con el objetivo)
    art_movible = O[i][art_actual]  # punto que se mueve para lograr el acercamiento
    
    # articulación rotatoria

    if(tipo_articulaciones[art_actual] == 0): 
      tita = obtener_angulo(objetivo, art_final, art_movible)
      if ((th[art_actual]+tita) > limites_max[art_actual]): 
        th[art_actual] = limites_max[art_actual]
      elif ((th[art_actual]+tita) < limites_min[art_actual]):
        th[art_actual] = limites_min[art_actual]
      else:
        th[art_actual] = th[art_actual]+tita

    # articulación prismática

    else: 
      w = np.sum(th[0: art_actual + 1])  # dirección w (suma de todos los ángulos relativos desde el inicio hasta la articulación que se mueve)
      w_vector_unitario = [cos(w), sin(w)]  # vector unitario con dirección w
      r_On = np.subtract(objetivo, art_final) # vector que va del punto final al objetivo
      d = np.dot(w_vector_unitario, r_On) # distancia que se debe extender la articulación (producto escalar)

      if ((a[art_actual]+d) > limites_max[art_actual]):
        a[art_actual] = limites_max[art_actual]
      elif ((a[art_actual]+d) < limites_min[art_actual]):
        a[art_actual] = limites_min[art_actual]
      else:
        a[art_actual] = a[art_actual]+d

    O[i+1] = cin_dir(th,a)

  dist = np.linalg.norm(np.subtract(objetivo,O[-1][-1]))
  print ("\n- Iteracion " + str(iteracion) + ':')
  muestra_origenes(O[-1])
  muestra_robot(O,objetivo)
  print ("Distancia al objetivo = " + str(round(dist,5)))
  iteracion+=1
  O[0]=O[-1]

if dist <= EPSILON:
  print("\n" + str(iteracion) + " iteraciones para converger.")
else:
  print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist,5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
  print("  theta" + str(i+1) + " = " + str(round(th[i],3)))
for i in range(len(th)):
  print("  L" + str(i+1) + "     = " + str(round(a[i],3)))
