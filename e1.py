import numpy as np
import pandas as p
import matplotlib.pyplot as plt

data=p.read_csv("D:/Java/ML/e1.csv")
data.head()
data.dropna(inplace=True)

x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

print(x)
print(y)

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtr,ytr)

predict=model.predict(xte)

plt.scatter(xtr,ytr,color='red')
plt.plot(xtr,model.predict(xtr),color='blue')
plt.title("Selling Price of Car by their Mileage Running")

plt.xlabel("Mileage")
plt.ylabel("Selling Price")
plt.show()
