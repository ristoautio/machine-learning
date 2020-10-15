import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('50_Startups.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(transformer.fit_transform(x))
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state=935765)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

# single prediction
print(lr.predict([[1, 0, 0, 160000, 130000, 300000]]))

# equation
coef = list(lr.coef_)
coef_frm = ['{:.2f} X v{}'.format(x, coef.index(x)) for x in coef]
coef_str = ' + '.join(coef_frm)

print('Profit = {} + {}'.format(coef_str, lr.intercept_))
