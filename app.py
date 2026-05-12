# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
class Linear_Regression():

    def __init__(self, Learning_rate, Numberof_Iteration):
        self.learning_rate = Learning_rate
        self.number_iteration = Numberof_Iteration

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = X
        self.y = Y

        for i in range(self.number_iteration):
            self.update_weights()

    def update_weights(self):

        y_predict = self.predict(self.x)

        dw = -(2 * (self.x.T).dot(self.y - y_predict)) / self.m
        db = -2 * np.sum(self.y - y_predict) / self.m

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, x):

        return x.dot(self.w) + self.b

model=pickle.load(open("linear.sav",'rb'))
print(model.predict(np.array([[1.5]])))
