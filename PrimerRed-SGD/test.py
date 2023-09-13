"""
Cargamos el código que sube lee los datos y leemos los 
training data, los datos de validación y los del test.
Preparamos los datos de entrenamiento en una vector
"""
import botella
training_data, validation_data, test_data = botella.load_data_wrapper()
training_data = list(training_data)


import network
"""
Aquí llamamos al código de la red neuronal y le damos los parámetros
con los que se va a entrenar la red.
"""
net = network.Network([784, 30, 10])
net.SGD(training_data, 20, 10, 0.07, test_data=test_data) 

