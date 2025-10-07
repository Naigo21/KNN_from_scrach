import numpy as np
import pandas as pd
from kNearestNeighbours import KNearestNeighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')

X=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

knn=KNearestNeighbors(3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)

print('accuracy_score : ' , accuracy_score(y_test, y_pred))






