# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:07:42 2017

@author: carbajal
"""

import numpy as np

X_train= np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
X_train= X_train[:,np.newaxis]
y_train=np.array([0, 0, 0, 1, 0, 1, 1, 1])
