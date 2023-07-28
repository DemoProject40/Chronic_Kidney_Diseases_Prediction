import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def Data_Preprocessing():
    df=pd.read_csv(os.path.join('data/raw','kidney_disease.csv'))
    print(df.head())
