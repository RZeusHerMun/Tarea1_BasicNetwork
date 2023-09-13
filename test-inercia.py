{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "da1de0fc",
   "metadata": {},
   "source": [
    "Esta prueba es para ver el rendimiento de la red neuronal solo con el **SGD**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f72b1648",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nbimporter \n",
    "import cargador\n",
    "training_data, validation_data, test_data = cargador.load_data_wrapper()\n",
    "training_data = list(training_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8d66a68",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "74517c12",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0 : 4113 / 10000\n",
      "Epoch 1 : 5438 / 10000\n",
      "Epoch 2 : 6745 / 10000\n",
      "Epoch 3 : 7203 / 10000\n",
      "Epoch 4 : 7436 / 10000\n",
      "Epoch 5 : 7589 / 10000\n",
      "Epoch 6 : 7695 / 10000\n",
      "Epoch 7 : 7776 / 10000\n",
      "Epoch 8 : 7835 / 10000\n",
      "Epoch 9 : 7891 / 10000\n",
      "Epoch 10 : 7941 / 10000\n",
      "Epoch 11 : 8306 / 10000\n",
      "Epoch 12 : 8594 / 10000\n",
      "Epoch 13 : 8653 / 10000\n",
      "Epoch 14 : 8693 / 10000\n",
      "Epoch 15 : 8736 / 10000\n",
      "Epoch 16 : 8776 / 10000\n",
      "Epoch 17 : 8811 / 10000\n",
      "Epoch 18 : 8842 / 10000\n",
      "Epoch 19 : 8861 / 10000\n",
      "Epoch 20 : 8890 / 10000\n",
      "Epoch 21 : 8904 / 10000\n",
      "Epoch 22 : 8923 / 10000\n",
      "Epoch 23 : 8937 / 10000\n",
      "Epoch 24 : 8955 / 10000\n",
      "Epoch 25 : 8973 / 10000\n",
      "Epoch 26 : 8980 / 10000\n",
      "Epoch 27 : 9001 / 10000\n",
      "Epoch 28 : 9010 / 10000\n",
      "Epoch 29 : 9022 / 10000\n",
      "Epoch 30 : 9028 / 10000\n",
      "Epoch 31 : 9043 / 10000\n",
      "Epoch 32 : 9041 / 10000\n",
      "Epoch 33 : 9059 / 10000\n",
      "Epoch 34 : 9073 / 10000\n",
      "Epoch 35 : 9078 / 10000\n",
      "Epoch 36 : 9084 / 10000\n",
      "Epoch 37 : 9096 / 10000\n",
      "Epoch 38 : 9107 / 10000\n",
      "Epoch 39 : 9113 / 10000\n",
      "Epoch 40 : 9119 / 10000\n",
      "Epoch 41 : 9125 / 10000\n",
      "Epoch 42 : 9133 / 10000\n",
      "Epoch 43 : 9132 / 10000\n",
      "Epoch 44 : 9148 / 10000\n",
      "Epoch 45 : 9155 / 10000\n",
      "Epoch 46 : 9147 / 10000\n",
      "Epoch 47 : 9158 / 10000\n",
      "Epoch 48 : 9165 / 10000\n",
      "Epoch 49 : 9168 / 10000\n"
     ]
    }
   ],
   "source": [
    "import nbimporter\n",
    "import red\n",
    "net = red.Network([784, 30, 10])\n",
    "net.SGD(training_data, 50, 10, 0.07, test_data=test_data)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b85ab222",
   "metadata": {},
   "source": [
    "La siguiente es la prueba de la red neuronal con el **SGD+Inercia**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "42650ff5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nbimporter \n",
    "import cargador\n",
    "training_data, validation_data, test_data = cargador.load_data_wrapper()\n",
    "training_data = list(training_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7698a379",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 0 : 4177 / 10000\n",
      "Epoch 1 : 5783 / 10000\n",
      "Epoch 2 : 7240 / 10000\n",
      "Epoch 3 : 7750 / 10000\n",
      "Epoch 4 : 8028 / 10000\n",
      "Epoch 5 : 8230 / 10000\n",
      "Epoch 6 : 8404 / 10000\n",
      "Epoch 7 : 8490 / 10000\n",
      "Epoch 8 : 8570 / 10000\n",
      "Epoch 9 : 8639 / 10000\n",
      "Epoch 10 : 8692 / 10000\n",
      "Epoch 11 : 8732 / 10000\n",
      "Epoch 12 : 8761 / 10000\n",
      "Epoch 13 : 8808 / 10000\n",
      "Epoch 14 : 8835 / 10000\n",
      "Epoch 15 : 8867 / 10000\n",
      "Epoch 16 : 8883 / 10000\n",
      "Epoch 17 : 8911 / 10000\n",
      "Epoch 18 : 8933 / 10000\n",
      "Epoch 19 : 8959 / 10000\n",
      "Epoch 20 : 8971 / 10000\n",
      "Epoch 21 : 8987 / 10000\n",
      "Epoch 22 : 9003 / 10000\n",
      "Epoch 23 : 9027 / 10000\n",
      "Epoch 24 : 9033 / 10000\n",
      "Epoch 25 : 9055 / 10000\n",
      "Epoch 26 : 9068 / 10000\n",
      "Epoch 27 : 9065 / 10000\n",
      "Epoch 28 : 9079 / 10000\n",
      "Epoch 29 : 9087 / 10000\n",
      "Epoch 30 : 9096 / 10000\n",
      "Epoch 31 : 9109 / 10000\n",
      "Epoch 32 : 9113 / 10000\n",
      "Epoch 33 : 9124 / 10000\n",
      "Epoch 34 : 9126 / 10000\n",
      "Epoch 35 : 9137 / 10000\n",
      "Epoch 36 : 9152 / 10000\n",
      "Epoch 37 : 9155 / 10000\n",
      "Epoch 38 : 9156 / 10000\n",
      "Epoch 39 : 9168 / 10000\n",
      "Epoch 40 : 9166 / 10000\n",
      "Epoch 41 : 9181 / 10000\n",
      "Epoch 42 : 9179 / 10000\n",
      "Epoch 43 : 9188 / 10000\n",
      "Epoch 44 : 9184 / 10000\n",
      "Epoch 45 : 9188 / 10000\n",
      "Epoch 46 : 9196 / 10000\n",
      "Epoch 47 : 9208 / 10000\n",
      "Epoch 48 : 9207 / 10000\n",
      "Epoch 49 : 9215 / 10000\n"
     ]
    }
   ],
   "source": [
    "import nbimporter\n",
    "import red\n",
    "net = red.Network([784, 30, 10])\n",
    "net.SGD(training_data, 50, 10, 0.07, test_data=test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "1ca9ec28",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Entrenamiento de la red neuronal...\n",
      "Epoch 0 : 1505 / 10000\n",
      "Epoch 1 : 2874 / 10000\n",
      "Epoch 2 : 4486 / 10000\n",
      "Epoch 3 : 5136 / 10000\n",
      "Epoch 4 : 6353 / 10000\n",
      "Time_training = 36.27275991439819 s.\n"
     ]
    }
   ],
   "source": [
    "import nbimporter \n",
    "import cargador\n",
    "training_data, validation_data, test_data = cargador.load_data_wrapper()\n",
    "training_data = list(training_data)\n",
    "'''--------------------------------------------------------------------------'''\n",
    "\n",
    "import nbimporter\n",
    "import red\n",
    "import time\n",
    "print(\"Entrenamiento de la red neuronal...\")\n",
    "inicio=time.time()\n",
    "net = red.Network([784, 30, 10])\n",
    "net.SGD(training_data, 5, 10, 0.07, test_data=test_data)\n",
    "fin=time.time()\n",
    "\n",
    "print(f'Time_training = {fin-inicio} s.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c0b5de9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
