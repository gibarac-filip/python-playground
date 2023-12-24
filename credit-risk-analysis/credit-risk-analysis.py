import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder

file_location = 'credit-risk-analysis/loan.csv'
data = pd.read_csv(file_location)
data.head()
data.info()

dictionary = pd.read_excel('credit-risk-analysis/loan-dictionary.xlsx')
dictionary = dictionary.dropna()

# Normalize numerical features
scaler = StandardScaler()
num_cols = ['loan_amount', 'interest_rate', 'income']
data[num_cols] = scaler.fit_transform(data[num_cols])

# Encode string features 
cat_cols = ['employment_status', 'home_ownership']
for col in cat_cols:
    encoder = LabelEncoder() 
    data[col] = encoder.fit_transform(data[col])

# Split features and target
X = data.drop('loan_status', axis=1)  
y = data['loan_status']

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(n_features):
    model = Sequential()
    model.add(Dense(10, input_dim=n_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

tf.get_logger().setLevel('ERROR')
# Feature extraction model
model = create_model(X_train.shape[1])

select = SelectKBest(f_classif, k=10)

pipeline = Pipeline([('select', select), ('model', model)])
    
# Evaluate cross validation
scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=metrics.make_scorer(accuracy_score))

print(f'Cross-validation accuracy: {np.mean(scores)}')