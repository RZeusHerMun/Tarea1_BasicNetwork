import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """En esta función: Definimos los atributos de la clase: numero de neuronas por capas, bias y weights"""
        self.num_layers = len(sizes)#numero de capas
        self.sizes = sizes# numero de neuronas
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]# Llenamos un vector con bias aleatorios.
        self.weights = [np.random.randn(y, x)# Tambien llenamos un vector de la matriz con los weights aleatorios.
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Esta función nos da el valor de inicialización 'a' determinandolo a partir de la f. sigmoide."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Esta función es el Stochastic Gradient Descent; recordando que basa su función en 
        obtener los weights adecuados con el back propagation, ademas de que calcula solo una derivada 
        por minibatch."""

        training_data = list(training_data)
        n = len(training_data)# Leemos los datos en forma de lista y los contamos.

        if test_data:# Aqui comenzamos el entrenamiento
            test_data = list(test_data)
            n_test = len(test_data)# Convertimos los datos de prueba y los contamos.

        for j in range(epochs):# Para el numero de epocas q definamos se entrenara como sigue.
            random.shuffle(training_data)#Barajeamos los datos de entrenamiento
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]#Dividimos los datos entre el numero de minibatchs.
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)# Ahora actualizamos los datos de los minibatchs ahora tambien integrando el LearningRate
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))# imprimimos el resultado de el entrenamiento con los datos del modelo.
            else:
                print("Epoch {} complete".format(j))# Este es por si no tenemos datos de prueba, para ver que van terminando las epocas.

    def update_mini_batch(self, mini_batch, eta):
        """Aqui definimos la actualizacion de los bias y los weights de los minibatchs"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights] # Usamos dos listas para almacenar las sumas de los gradientes de cada minibatch. Son del mismo tamano que las de las bias y los weights.
        for x, y in mini_batch:# En este ciclo usamos la funcion del BackPropagation definida mas abajo en la que calculamos los gradientes de C (la funcion de coste) con los bias y weights 'originales'. Despues sumamos a los gradientes definimos (los nabla)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]#Cuando se calculan todos los gradientes del minibatch, se actualizan los bias y los weights con la resta del aprendizaje.

    def backprop(self, x, y):
        """Codigo del Algoritmo BackPropagation"""
        nabla_b = [np.zeros(b.shape) for b in self.biases] # Creamos las listas con los bias y los weights determinados.
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x] # Esta será la lista con todas las activaciones de las capas de la red. El primer valor es x
        zs = [] # esta es la lista (vacia por el momento) para las sumas de los pesos de cada capa
        for b, w in zip(self.biases, self.weights): #Este ciclo es para cada cada capa de la red.
            z = np.dot(w, activation)+b# CAlculamos la 'a' para sacar la suma ponderada
            zs.append(z) # Vamos agregando cada valor a la lista
            activation = sigmoid(z) # Se calcula el valor de la f sigmoide con la a calculada
            activations.append(activation) # la agregamos a la lista
        
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])# una vez tenemos la lista de las activaciones, sacamos el error de la ultima capa.
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # guardamos los errores en las listas "nablas" de atras hacia adelante.

        for l in range(2, self.num_layers): #ahora usamos los errores para calcular los gradientes de la ultima capa.
            z = zs[-l] #empezamos desde la capa anterior
            sp = sigmoid_prime(z) #definimos la derivada
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #calculamos el error en esa capa usando del error de la capa siguiente
            nabla_b[-l] = delta# usamos el error para calcular los gradientes de la capa y los vamos guardando en las listas de "nablas".
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)# al final las listas ya tiene los gradientes de cada capa de la red.

    def evaluate(self, test_data):
        """En esta función evaluamos los datos de prueba que clasifico la red."""
        test_results = [(np.argmax(self.feedforward(x)), y)# en esta lista se encuentran los resultados de la clasificación. Buscamos el argumento maximo de la función feedforward para calcular la salida de la red.
                        for (x, y) in test_data]# el proceso se hace en cada valor donde tenemos el resultado verdadero y el valor predicho. Esto se almacena en un vector de dos columnas.
        return sum(int(x == y) for (x, y) in test_results)# Presentamos la lista de resultados

    def cost_derivative(self, output_activations, y):
        """ Definimos la funcion de costo con la derivada respecto a las activaciones de salida."""
        return (output_activations-y)


def sigmoid(z):
    """Esta es la funcion sigmoide que usaremos"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Y aqui­ definimos la derivada de la funcion sigmoide."""
    return sigmoid(z)*(1-sigmoid(z))