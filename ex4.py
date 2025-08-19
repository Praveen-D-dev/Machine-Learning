import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

data=pd.read_csv("ex4.csv")
data.head()
df=pd.DataFrame(data)
x=df[["study_hours", "attendance"]]
y=df["result"]
clf=DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(x, y)

plt.figure(figsize=(8, 6))
plot_tree(clf, filled=True, feature_names=["study_hours", "attendance"], class_names=["1","0"], fontsize=10)
plt.show()
new=[[2,65]]
pred=clf.predict(new)
print("Prediction for pass of fail:","1" if pred[0] == 1 else "0")