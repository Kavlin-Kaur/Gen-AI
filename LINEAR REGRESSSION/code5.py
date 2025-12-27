from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([[1],[2],[3]])
y = np.array([3, 12, 27])

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression().fit(x_poly, y)
print(model.predict(x_poly))
