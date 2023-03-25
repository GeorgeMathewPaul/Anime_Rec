import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv('music.csv')
X=df.drop(columns=['genre'])
y=df['genre']
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2)
#Model is created
model=DecisionTreeClassifier()
#Fit to train the model
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(X_test)
print (predictions)
