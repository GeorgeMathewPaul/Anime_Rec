import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('music.csv')
X=df.drop(columns=['genre'])
y=df['genre']
#Model is created
model=DecisionTreeClassifier()
#Fit to train the model
model.fit(X,y)
predictions = model.predict([[21,1],[22,0]])
print (predictions)