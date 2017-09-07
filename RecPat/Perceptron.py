import random

class Perceptron:
    def __init__(self,input_number,step_size=0.1):
    self._ins = input_number # Número de parámetros de entrada
    
    # Seleccionamos pesos aleatorios
    self._w = [random.random() for _ in range(input_number)]
    self._eta = step_size # La tasa de convergencia
    
    def predict(self,inputs):
        # Producto punto de entrada y pesos
        weighted_average = sum(w*elm for w,elm in zip(self._w,inputs))
        if weighted_average > 0:
            return 1
        return 0
    
    def train(self,inputs,ex_output):
        output = self.predict(inputs)
        error = ex_output - output
        # El error es la diferencia entre la salida correcta y la esperada
        if error != 0:
            self._w = [w+self._eta*error*x for w,x in
            zip(self._w,inputs)]
        return error
    
#!/usr/bin/env python
from perceptron import Perceptron

## Datos de hombres y mujeres
input_data = [[1.7,56,1], # Mujer de 1.70m y 56kg
              [1.72,63,0],# Hombre de 1.72m y 63kg
              [1.6,50,1], # Mujer de 1.60m y 50kg
              [1.7,63,0], # Hombre de 1.70m y 63kg
              [1.74,66,0],# Hombre de 1.74m y 66kg
              [1.58,55,1],# Mujer de 1.58m y 55kg
              [1.83,80,0],# Hombre de 1.83m y 80kg
              [1.82,70,0],# Hombre de 1.82m y 70kg
              [1.65,54,1]]# Mujer de 1.65m y 54kg

## Creamos el perceptron
pr = Perceptron(3) # Perceptron con 3 entradas

## Fase de entrenamiento
for _ in range(100):
    # Vamos a entrenarlo varias veces sobre los mismos datos
    # para que los 'pesos' converjan
    for person in input_data:
        output = person[-1]
        inp = [1] + person[0:-1] # Agregamos un uno por default
        err = pr.train(inp,output)

h = float(raw_input("Introduce tu estatura en centimetros.- "))
w = float(raw_input("Introduce tu peso en kilogramos.- "))

if pr.predict([1,h,2]) == 1: print "Mujer"
else: print "Hombre"
