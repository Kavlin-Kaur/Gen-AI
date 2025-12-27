from sklearn.linear_model import LogisticRegression
import numpy as np

x = np.array([[1],[2],[3],[4]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression().fit(x, y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", model.predict(x))
