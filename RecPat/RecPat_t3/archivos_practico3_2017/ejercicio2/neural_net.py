# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:33:35 2017

@author: recpat
"""

import numpy as np
import sys


class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.
    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatche_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.
    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.
    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation
        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.
        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)
        """
        onehot = np.zeros((n_classes, y.shape[0]),dtype=int)
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def _softmax(self, z):
        """Compute logistic function (sigmoid)"""
        exp_z = np.exp(z)
        suma_exp_z = np.sum(exp_z,axis=1)
        return exp_z / suma_exp_z[:,np.newaxis]    

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        S_h = np.dot(X, self.W_h) + self.b_h

        # step 2: activation of hidden layer
        Y = self._sigmoid(S_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        S_out = np.dot(Y, self.W_out) + self.b_out

        # step 4: activation output layer
        Z = self._sigmoid(S_out)
        #Z = self._softmax(S_out)

        return S_h, Y, S_out, Z

    def _compute_cost(self, y_enc, output):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)
        Returns
        ---------
        cost : float
            Regularized cost
        """
        L2_term = (self.l2 *
                   (np.sum(self.W_h ** 2.) +
                    np.sum(self.W_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """Predict class labels
        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """
        S_h, Y, S_out, Z = self._forward(X)
        y_pred = np.argmax(S_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train):
        """ Learn weights from training data.
        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        Returns:
        ----------
        self
        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.W_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                S_h, Y, S_out, Z = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                dS_out = Z - y_train_enc[batch_idx]

                # -----------------------------------------------------# 
                #                INICIO COMPLETAR
                # -----------------------------------------------------#
  

                #  [n_features, n_hidden]  
                grad_W_h = np.zeros((n_features, self.n_hidden))   
                #  [n_hidden]
                grad_b_h = np.zeros(self.n_hidden)  		  

                #  [n_hidden, n_classlabels]
                grad_W_out = np.zeros((self.n_hidden, n_output))  
                #  [n_classlabels]
                grad_b_out = np.zeros(n_output)                    
 
               
                # -----------------------------------------------------#
                #                FIN COMPLETAR
                # -----------------------------------------------------# 

                # Regularization and weight updates
                delta_W_h = (grad_W_h + self.l2*self.W_h)
                delta_b_h = grad_b_h  # bias is not regularized
                self.W_h -= self.eta * delta_W_h
                self.b_h -= self.eta * delta_b_h

                delta_W_out = (grad_W_out + self.l2*self.W_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.W_out -= self.eta * delta_W_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each N epochs during training
            if i % 10 == 0:
                
                S_h, Y, S_out, Z = self._forward(X_train)
                cost = self._compute_cost(y_enc=y_train_enc,
                                      output=Z)

                y_train_pred = self.predict(X_train)

                train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])

                sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train Acc.: %.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100))
                sys.stderr.flush()

                self.eval_['cost'].append(cost)
                self.eval_['train_acc'].append(train_acc)

        return self
