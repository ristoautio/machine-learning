import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
y = y.reshape(len(y), 1)

sc = StandardScaler()
sc_y = StandardScaler()
x = sc.fit_transform(x)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(x,y)

plt.scatter(sc.inverse_transform(x), sc_y.inverse_transform(y),color='red')
plt.plot(sc.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color='blue')
plt.show()