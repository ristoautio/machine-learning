import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# replace missing values with the average
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])

# split country column into separate columns
transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(transformer.fit_transform(x))

# yes/no to 1/0
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# split to training and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state=399536)

# scale features
scaler = StandardScaler()
x_train[:, 3:] = scaler.fit_transform(x_train[:,3:])
x_test[:, 3:] = scaler.fit_transform(x_test[:,3:])
