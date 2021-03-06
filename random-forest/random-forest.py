import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

regressor = RandomForestRegressor(random_state=0, n_estimators=10)
regressor.fit(x_train, y_train)
res = regressor.predict(x_test)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid))
plt.show()
