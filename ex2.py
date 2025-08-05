import pandas as p
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data=p.read_csv("D:/Java/ML/e2.csv")
data.head()
data.dropna(inplace=True)

x=data[['Bedrooms','Size','Years','ZipCode']]
y=data['Selling-Price']

print(x)
print(y)

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),['ZipCode'])],remainder='passthrough')
xencode=ct.fit_transform(x)
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(xtr,ytr)

predict=model.predict(xte)

coefficient=model.coef_
intercept=model.intercept_
print("Coefficent",coefficient)
print("Intercept",intercept)
plt.figure(figsize=(8,6))
sns.scatterplot(x=yte,y=predict,color='blue',s=100)
plt.plot([min(yte),max(yte)],[min(yte),max(yte)],'r--')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual VS Predictied House Price')
plt.grid(True)
plt.tight_layout()
plt.show()
sns.heatmap(x.corr(), annot=True,cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
