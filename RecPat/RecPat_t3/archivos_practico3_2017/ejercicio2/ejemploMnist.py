# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:44:43 2017

@author: carbajal
"""

from utils import load_mnist_unziped

X_train, y_train = load_mnist_unziped('mnist', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist_unziped('mnist', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


import matplotlib.pyplot as plt

plt.close('all')

# Se deberían ver 2 × 5 subfiguras con una imagen  
# representativa de cada dígito

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
  img = X_train[y_train == i][0].reshape(28, 28)
  ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Se deberían ver 25 imágenes del dígito seleccionado

digito = 5
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
  img = X_train[y_train == digito][i].reshape(28, 28)
  ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

from neural_net import NeuralNetMLP

nn = NeuralNetMLP(n_hidden=50, l2=0.1, epochs=100, 
                  eta=0.001, shuffle=True, minibatch_size=50, seed=1)
                                  

nn.fit(X_train, y_train)

import numpy as np

y_train_pred = nn.predict(X_train)
acc = float( np.sum(y_train == y_train_pred, axis=0)) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))    

y_test_pred = nn.predict(X_test)
acc = float(np.sum(y_test == y_test_pred, axis=0)) / X_test.shape[0]
print('Testing accuracy: %.2f%%' % (acc * 100))             