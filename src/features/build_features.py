import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add the path to the logger file (outer folder) to the Python path
logging_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(logging_path)

# Now you can import the logger module from the "src" folder
from logger import logging


def Data_Preprocessing():

    logging.info("Data Ingestion started...")

    #Load the data 
    df=pd.read_csv(os.path.join('data_files/raw','kidney_disease.csv'))

    logging.info("Train test split sucessfull..")
    #Split the data into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    #Create the "data_folder" if it doesn't exist
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)


    #Create the "artifacts" folder inside "data_folder" if it doesn't exist
    artifacts_folder = os.path.join(data_folder, "artifacts")
    if not os.path.exists(artifacts_folder):
        os.makedirs(artifacts_folder)

    #Save the train and test data in the "artifacts" folder
    train_data.to_csv(os.path.join(artifacts_folder, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(artifacts_folder, "test_data.csv"), index=False)

    logging.info("Data ingestion Completed...")

    return train_data.shape,test_data.shape

print(Data_Preprocessing())

    


    
