import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

model=joblib.load('music-recommender.joblib')
predictions =model.predict([[21,1]])
print(predictions)