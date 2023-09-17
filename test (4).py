#!/usr/bin/env python
# coding: utf-8

# ### Esta prueba es para ver el rendimiento de la red neuronal solo con el **SGD**.

# In[ ]:


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


# In[17]:


t_sgd = fin-inicio
t_sgd=round(t_sgd, 4)


# ### Prueba de Rendimiento con la red optimizada con *SGD+Inercia 

# In[20]:


import nbimporter 
import cargador
training_data, validation_data, test_data = cargador.load_data_wrapper()
training_data = list(training_data)

import nbimporter
import red
import time
print("Entrenamiento de la red neuronal con SGD+Inercia...")
inicio=time.time()
net = red.Network([784, 30, 10])
net.SGD(training_data, 15, 10, 0.07, test_data=test_data)
fin=time.time()
print(f'Tiempo de Entrenamiento = {fin-inicio} s.')
t_inercia = fin-inicio
t_inercia=round(t_inercia, 4)


# In[3]:


import matplotlib.pyplot as plt
epochs = list(range(15))
v_SGD = [3761, 4738, 5392, 6285, 6521, 6715, 6860, 6953, 7049, 7094, 7142, 7190, 7222, 7245, 7260]
v_SGDin= [2807, 4884, 6140, 6898, 7270, 7533, 7678, 7775, 7885, 7944, 7999, 8062, 8144, 8389, 8541]
t_sgd = 135.5005
t_inercia = 97.9238
plt.figure(figsize=(10,6))
plt.plot(epochs, v_SGD, label=f'SGD = {t_sgd} s')
plt.plot(epochs, v_SGDin, label=f'SGD+Inercia = {t_inercia} s')
plt.xlabel('Epocas')
plt.ylabel('Aciertos')
plt.ylim([0,10000])
plt.title('Efectividad de RNA con SGD y SGD+Inercia por época')
plt.legend()
plt.show()


# ### Este fue para probar la red (sgd+inercia) con mu=0.5

# In[23]:


import nbimporter 
import cargador
training_data, validation_data, test_data = cargador.load_data_wrapper()
training_data = list(training_data)

import nbimporter
import red
import time
print("Entrenamiento de la red neuronal con SGD+Inercia...")
inicio=time.time()
net = red.Network([784, 30, 10])
net.SGD(training_data, 15, 10, 0.07, 0.9, test_data=test_data)#(epocas, minibatch, eta, mu)
fin=time.time()
print(f'Tiempo de Entrenamiento = {round(fin-inicio, 7)} s')
t_inercia = fin-inicio
t_inercia=round(t_inercia, 4)


# #### Este fue para probar la red (sgd+inercia) con mu=0.9

# In[27]:


import nbimporter 
import cargador
training_data, validation_data, test_data = cargador.load_data_wrapper()
training_data = list(training_data)

import nbimporter
import red
import time
print("Entrenamiento de la red neuronal con SGD+Inercia...")
inicio=time.time()
net = red.Network([784, 30, 10])
net.SGD(training_data, 15, 10, 0.07, test_data=test_data)#(epocas, minibatch, eta) no le pude meter el mu aqui
fin=time.time()
print(f'Tiempo de Entrenamiento = {round(fin-inicio, 7)} s')
t_inercia = fin-inicio
t_inercia=round(t_inercia, 4)


# In[44]:


import matplotlib.pyplot as plt
epochs = list(range(15))
v_SGD = [3761, 4738, 5392, 6285, 6521, 6715, 6860, 6953, 7049, 7094, 7142, 7190, 7222, 7245, 7260]
v_SGDin= [2807, 4884, 6140, 6898, 7270, 7533, 7678, 7775, 7885, 7944, 7999, 8062, 8144, 8389, 8541]
v_SGDin_05 = [3407, 5013, 5493, 5790, 6478, 6745, 6876, 6957, 7029, 7079, 7131, 7176, 7203, 7231, 7250]
v_SGDin_09 = [4286, 6541, 7570, 7967, 8207, 8361, 8495, 8567, 8622, 8683, 8727, 8784, 8805, 8834, 8845]

pct_SGD = round((v_SGD[-1]*100)/10000 ,2)
pct_SGDin = round((v_SGDin[-1]*100)/10000 ,2)
pct_SGDin_05 = round((v_SGDin_05[-1]*100)/10000 ,2)
pct_SGDin_09 = round((v_SGDin_09[-1]*100)/10000 ,2)

plt.figure(figsize=(10,6))

plt.plot(epochs, v_SGD, label=f'SGD = {t_sgd} s | ({pct_SGD}%)')
plt.plot(epochs, v_SGDin, label=f'SGD+Inercia (mu=???) = 97.9237 s | ({pct_SGDin}%)')
plt.plot(epochs, v_SGDin_05, label=f'SGD+Inercia (mu=0.5) = 126.9520 s | ({pct_SGDin_05}%)')
plt.plot(epochs,v_SGDin_09,label=f'SGD+Inercia (mu=0.9) = {t_inercia09} s | ({pct_SGDin_09}%)')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,10000])
plt.title('Efectividad de RNA con SGD y SGD+Inercia por época')
plt.legend()
plt.show()


# ### Entrenamiento con el RMSprop
# 

# In[9]:


import nbimporter 
import cargador
training_data, validation_data, test_data = cargador.load_data_wrapper()
training_data = list(training_data)

import nbimporter
import red
import time
print("Entrenamiento de la red neuronal con RMSprop...")
inicio=time.time()
net = red.Network([784, 30, 10])
net.SGD(training_data, 20, 15, 0.0001, test_data=test_data)
fin=time.time()
print(f'Tiempo de Entrenamiento = {round(fin-inicio, 7)} s')
t_inercia = fin-inicio
t_inercia=round(t_inercia, 4)


# In[42]:


import matplotlib.pyplot as plt
v_SGD = [3761, 4738, 5392, 6285, 6521, 6715, 6860, 6953, 7049, 7094, 7142, 7190, 7222, 7245, 7260]
v_SGDin= [2807, 4884, 6140, 6898, 7270, 7533, 7678, 7775, 7885, 7944, 7999, 8062, 8144, 8389, 8541]
v_SGDin_05 = [3407, 5013, 5493, 5790, 6478, 6745, 6876, 6957, 7029, 7079, 7131, 7176, 7203, 7231, 7250]
v_SGDin_09 = [4286, 6541, 7570, 7967, 8207, 8361, 8495, 8567, 8622, 8683, 8727, 8784, 8805, 8834, 8845]
v_RMSprop = [9072, 9208, 9219, 9265, 9286, 9263, 9358, 9286, 9348, 9369, 9436, 9413, 9413, 9394, 9415, 9423, 9427, 9438, 9407, 9462, 9416, 9469, 9483, 9463, 9439, 9482, 9474, 9454, 9470, 9452]

pct_SGD = 72.60
pct_SGDin = 85.41
pct_SGDin_05 = 72.50
pct_SGDin_09 = 72.50
pct_RMSprop = round((v_RMSprop[-1]*100)/10000 ,2)

epochs = list(range(1,len(v_RMSprop)+1))
# debo hacer un ajuste para que las epochs los de el vector más largo y de ahí sacar la longitud deseada.
long = 30 
v_SGD.extend([None] * (long - len(v_SGD)))
v_SGDin.extend([None] * (long - len(v_SGDin)))
v_SGDin_05.extend([None] * (long - len(v_SGDin_05)))
v_SGDin_09.extend([None] * (long - len(v_SGDin_09)))

plt.figure(figsize=(10,6))

plt.plot(epochs, v_SGD, label=f'SGD = 135.5005 s | ({pct_SGD}%)')
plt.plot(epochs, v_SGDin, label=f'SGD+Inercia (mu=???) = 97.9237 s | ({pct_SGDin}%)')
plt.plot(epochs, v_SGDin_05, label=f'SGD+Inercia (mu=0.5) = 126.9520 s | ({pct_SGDin_05}%)')
plt.plot(epochs,v_SGDin_09,label=f'SGD+Inercia (mu=0.9) = 135.4419 s | ({pct_SGDin_09}%)')
plt.plot(epochs,v_RMSprop,label=f'RMSprop (mu=0.9) = 251.9460 s | ({pct_RMSprop}%)')

plt.xlabel('Epocas')
plt.ylabel('Precisión')
plt.ylim([0,10000])
plt.title('Efectividad de RNA con distintos métodos de optimización')
plt.legend()
plt.show()


# In[43]:


import nbimporter 
import cargador
training_data, validation_data, test_data = cargador.load_data_wrapper()
training_data = list(training_data)

import nbimporter
import red
import time
print("Entrenamiento de la red neuronal con RMSprop...")
inicio=time.time()
net = red.Network([784, 30, 10])
net.SGD(training_data, 15, 5, 0.0001, test_data=test_data)
fin=time.time()
print(f'Tiempo de Entrenamiento = {round(fin-inicio, 7)} s')
t_inercia = fin-inicio
t_inercia=round(t_inercia, 4)


# In[49]:


import matplotlib.pyplot as plt
v_SGD = [3761, 4738, 5392, 6285, 6521, 6715, 6860, 6953, 7049, 7094, 7142, 7190, 7222, 7245, 7260]
v_SGDin= [2807, 4884, 6140, 6898, 7270, 7533, 7678, 7775, 7885, 7944, 7999, 8062, 8144, 8389, 8541]
v_SGDin_05 = [3407, 5013, 5493, 5790, 6478, 6745, 6876, 6957, 7029, 7079, 7131, 7176, 7203, 7231, 7250]
v_SGDin_09 = [4286, 6541, 7570, 7967, 8207, 8361, 8495, 8567, 8622, 8683, 8727, 8784, 8805, 8834, 8845]
v_RMSprop = [9054, 9144, 9289, 9304, 9374, 9399, 9372, 9333, 9392, 9446, 9450, 9421, 9466, 9423, 9448]

pct_SGD = 72.60
pct_SGDin = 85.41
pct_SGDin_05 = 72.50
pct_SGDin_09 = 72.50
pct_RMSprop = round((v_RMSprop[-1]*100)/10000 ,2)

epochs = list(range(1,len(v_RMSprop)+1))
# debo hacer un ajuste para que las epochs los de el vector más largo y de ahí sacar la longitud deseada.
'''
long = 30 
v_SGD.extend([None] * (long - len(v_SGD)))
v_SGDin.extend([None] * (long - len(v_SGDin)))
v_SGDin_05.extend([None] * (long - len(v_SGDin_05)))
v_SGDin_09.extend([None] * (long - len(v_SGDin_09)))
'''
plt.figure(figsize=(10,6))

plt.plot(epochs, v_SGD, label=f'SGD = 135.5005 s | ({pct_SGD}%)')
plt.plot(epochs, v_SGDin, label=f'SGD+Inercia (mu=???) = 97.9237 s | ({pct_SGDin}%)')
plt.plot(epochs, v_SGDin_05, label=f'SGD+Inercia (mu=0.5) = 126.9520 s | ({pct_SGDin_05}%)')
plt.plot(epochs,v_SGDin_09,label=f'SGD+Inercia (mu=0.9) = 135.4419 s | ({pct_SGDin_09}%)')
plt.plot(epochs,v_RMSprop,label=f'RMSprop (mu=0.9) = 125.7263 s | ({pct_RMSprop}%)')

plt.xlabel('Epocas')
plt.ylabel('Precisión')
plt.ylim([0,10000])
plt.title('Efectividad de RNA con distintos métodos de optimización')
plt.legend()
plt.show()


# In[ ]:




