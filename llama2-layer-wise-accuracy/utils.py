import pandas as pd

import os

def save_csv(df, path):
    df.to_csv(path, index=False)
    
def load_csv(path):
    return pd.read_csv(path)

def check_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_df(schema, path, filename):
    if not os.path.exists(path + filename):
        return pd.DataFrame(schema)
    else:
        return load_csv(path + filename) 

