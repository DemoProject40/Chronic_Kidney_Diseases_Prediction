import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def Model_Training():
    #load the dataset
    data = pd.read_csv(os.path.join('data_files/processed','new_model.csv'),sep = '\t')

    #spilit the data into X and Y
    x = data.drop(['Class'],axis=1)
    y = data['Class']
    lab_enc=LabelEncoder()
    y=lab_enc.fit_transform(y)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    #logging.info('Model Reportstarted')
    models = {
                'KNN': KNeighborsClassifier(),
                'DecisionTree' : DecisionTreeClassifier(),
                'Random_forest' : RandomForestClassifier(),
                'Naive_bayes' : GaussianNB(),
            }
    
    # Training and using the models
    for model_name, model in models.items():
        # Train the model on your data
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model (you might use some evaluation metrics here)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{model_name} accuracy: {accuracy}')

Model_Training()