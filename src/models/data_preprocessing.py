import os
import sys

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

# Add the path to the logger file (outer folder) to the Python path
logging_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(logging_path)

# Now you can import the logger module from the "src" folder
from logger import logging

def Data_Preprocessing():

    logging.info("Data preprocessing started...")

    #load the dataset
    df = pd.read_csv(os.path.join('data_files/raw','kidney_disease.csv'))

    #perform simplerImputer 
    imp_mode = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')

    data = pd.DataFrame(imp_mode.fit_transform(df))

    data.columns = df.columns
    
    logging.info(data.head())

    # Perform on the train data
    data['cad'] = data['cad'].apply(lambda x:'no' if x == "\tno" else x )

    data['dm'] = data['dm'].apply(lambda x:'no' if x == "\tno" else x )
    data['dm'] = data['dm'].apply(lambda x:'yes' if x == "\tyes" else x )
    data['dm'] = data['dm'].apply(lambda x:'yes' if x == " yes" else x )

    data['rc'] = data['rc'].apply(lambda x:'5.2' if x == "\t?" else x )

    data['wc'] = data['wc'].apply(lambda x:'9800' if x == "\t6200" else x )
    data['wc'] = data['wc'].apply(lambda x:'9800' if x == "\t8400" else x )
    data['wc'] = data['wc'].apply(lambda x:'9800' if x == "\t?" else x )

    data['pcv'] = data['pcv'].apply(lambda x:'41' if x == "\t43" else x )
    data['pcv'] = data['pcv'].apply(lambda x:'41' if x == "\t?" else x )

    data['classification'] = data['classification'].apply(lambda x:'ckd' if x == "ckd\t" else x )

    _df = data.apply(preprocessing.LabelEncoder().fit_transform)

    #_df.to_csv('../data/processed/final_data.csv')

    logging.info("Data preprocessing completed...")

Data_Preprocessing()