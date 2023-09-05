"""
Este es el código que va a cargar y leer los datos con los que 
disponemos.
"""

import pickle
import gzip

import numpy as np

def load_data(): # Esta función desencripta los datos, los organiza en listas y los codifica en un formato específico. 
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper(): # Con esta función vamos a organizar nuestros datos en las matrices que necesitamos
    tr_d, va_d, te_d = load_data() # usamos diminutivos para las listas
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] # Creamos una matriz de 2*784, para organizar los TrainigData. En una columna están el pixel de entrada y en otro su valor leido del mnist.
    training_results = [vectorized_result(y) for y in tr_d[1]] # Creamos el vector de las neuronas de salida.
    training_data = zip(training_inputs, training_results) # unimos las dos listas que se crearon en los pasos anteriores.
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]] # creamos la matriz donde van los datos de validación de entrada igual que los TrainingData.
    validation_data = zip(validation_inputs, va_d[1]) # Le ponemos en la segunda columna, los datos leídos del mnist.
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]] # lo mismo para los datos de prueba que en los pasos anteriores.
    test_data = zip(test_inputs, te_d[1])
    
    return (training_data, validation_data, test_data) #damos las matrices creadas.


def vectorized_result(j): # Establecemos una función que nos dé el vector donde se alojarán los números del 0 al 9, que serán las neuronas de salida.
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
