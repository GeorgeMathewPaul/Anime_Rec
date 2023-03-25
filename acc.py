import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('music.csv')
X=df.drop(columns=['genre'])
y=df['genre']
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.1)
#Model is created
model=DecisionTreeClassifier()
#Fit to train the model
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(X_test)
print (predictions)
score =accuracy_score(y_test,predictions)
print(score)
