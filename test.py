#!/usr/bin/env python
# coding: utf-8

# Esta prueba es para ver el rendimiento de la red neuronal solo con el **SGD**.

# In[2]:


import nbimporter 
import cargador
training_data, validation_data, test_data = cargador.load_data_wrapper()
training_data = list(training_data)

import nbimporter
import red
import time
print("Entrenamiento de la red neuronal...")
inicio=time.time()
net = red.Network([784, 30, 10])
net.SGD(training_data, 15, 10, 0.07, test_data=test_data)
fin=time.time()
print(f'Tiempo de Entrenamiento = {fin-inicio} s.')


# In[9]:


import matplotlib.pyplot as plt
epochs = list(range(15))
v_SGD = [3761, 4738, 5392, 6285, 6521, 6715, 6860, 6953, 7049, 7094, 7142, 7190, 7222, 7245, 7260]
#v_SGDin
plt.figure(figsize=(10,6))
plt.plot(epochs, v_SGD, label=f't (SGD) = {fin-inicio} s.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,10000])
plt.title('Efectividad de RNA con SGD por época')
plt.legend()
plt.show()


# In[ ]:




