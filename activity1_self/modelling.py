import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('house_rent.csv')

X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[1, 1, 2]]))
