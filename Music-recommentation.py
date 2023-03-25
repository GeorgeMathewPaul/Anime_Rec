import pandas as pd
# sklearn package, tree module, Class DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# below library for spliting data into train and test
from sklearn.model_selection import train_test_split
# Below library to claculate accuracy
from sklearn.metrics import accuracy_score
df=pd.read_csv('d:/ml/music.csv')
X=df.drop(columns=['genre'])
y=df['genre']
# Allocate 20% of data for testing. Usually 70-80% for training. This function return a tuple
# Split returns a tuple so we unpack it
# First 2 variables are input set for training
# Other 2 variables are output set for training
# If you use 0.8 for testing accuracy will drop
# Clean data remove duplicates relevant data remove null or fill with default etc more data you get 
# better result
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2)
model=DecisionTreeClassifier()
# fit takes two parameters input set and output set
# Instaed of passing full data 
# model.fmodel.fit(X,y)
# For training we pass only training data
model.fit(X_train,y_train)
# predict takes parameter as 2 dimensional array. Each row is input data
# For predicting we pass test data instead of constant data
# predictions = model.predict([[21,1],[22,0]])
predictions = model.predict(X_test)
# accuracy score take 2 arguments. First argument is Expected values or y_test. 
# Second argument is actual values or predictions
# When you run each score will be diffrent because train test split is random
score =accuracy_score(y_test,predictions)
score

# We will not train model always. Training a model is time consuming (usually millions of samples)
# So model persistency is important
# Once in  a while we build and trai a model and save it to a file
# Next time when we want to predict we load model from a file and sak to predict
# Model is already trained. We dont need to retrain it. Its like a intelligent person
# joblib have method to saving and loading models

from sklearn.externals import joblib

model =DecisionTreeClassifier(X,y)
model.fit(X,y)
# model persistence
# dump have 2 parameters namof the model and physical  file to dump. Its a binary file
joblib.dump(model,'music-recommender.joblib')

# only below line needed on run time
model=joblib.load('music-recommender.joblib')
predictions =model.predict([[21,1]])

# Visualize decision tree
from sklearn import tree
model =DecisionTreeClassifier()
model.fit(X,y)
# first argument is model
# second argument is name of output file
# keyword argument no need to worry about order of argument
# dot is graph description language
# third argument is feature_names these are array of column names of input data
# These are properties or features of our data
# labels of our output data
# open dot file using vscode
# install extension search for dot . Graphviz (dot) language Stephanvs
# VScode to right ... open preview to the side. Close dot file
# its a binary tree. Every node have max 2 children

# tree box rounded corners  true, filled true for box color, label all for every node have labels
tree.export_graphviz(model,out_file='music-recommender.dot',
                        feature_names=['áge','gender'],
                        class_names=sorted(y.unique()),
                         label='all',
                         rounded=True,
                        filled=True)
