import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

lr = LinearRegression()
lr.fit(x, y)

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lr_2 = LinearRegression()
lr_2.fit(x_poly, y)

plt.scatter(x,y,color='red')
plt.plot(x, lr.predict(x), color='blue')
plt.plot(x, lr_2.predict(x_poly), color='green')

print(lr.predict([[6.5]]))
print(lr_2.predict(poly_reg.fit_transform([[6.5]])))
plt.show()


