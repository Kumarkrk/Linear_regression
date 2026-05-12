# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
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



model = pickle.load(open("linear.sav", "rb"))

st.title("Salary Prediction App")

experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    step=0.1
)

if st.button("Predict Salary"):

    prediction = model.predict(np.array([[experience]]))

    st.success(f"Predicted Salary: ₹ {prediction[0]:.2f}")
